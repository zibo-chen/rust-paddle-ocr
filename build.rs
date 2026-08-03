use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

mod build_support;

use build_support::{
    cpp_runtime_libraries, cuda_side_library_plan, prebuilt_asset_name, select_link_mode,
    should_link_mnn_whole_archive, uses_msvc_flags, BuildFeatures, MnnLinkMode, NativeLinkKind,
    TargetInfo,
};

/// MNN prebuilt version to download from GitHub releases
const MNN_PREBUILT_VERSION: &str = "dev";
const MNN_PREBUILT_REPO: &str = "zibo-chen/MNN-Prebuilds";

fn main() {
    // 在 docs.rs 构建环境中，跳过所有 C++ 编译
    if env::var("DOCS_RS").is_ok() || env::var("CARGO_FEATURE_DOCSRS").is_ok() {
        println!("cargo:warning=Building for docs.rs, skipping C++ compilation");
        return;
    }

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let target_triple = env::var("TARGET").unwrap();
    let debug = env::var("DEBUG").unwrap();

    // Feature flags
    let coreml_enabled = env::var("CARGO_FEATURE_COREML").is_ok();
    let metal_enabled = env::var("CARGO_FEATURE_METAL").is_ok();
    let cuda_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();
    let opencl_enabled = env::var("CARGO_FEATURE_OPENCL").is_ok();
    let opengl_enabled = env::var("CARGO_FEATURE_OPENGL").is_ok();
    let vulkan_enabled = env::var("CARGO_FEATURE_VULKAN").is_ok();

    let mnn_dynamic = env::var("CARGO_FEATURE_MNN_DYNAMIC").is_ok();
    let mnn_static = env::var("CARGO_FEATURE_MNN_STATIC").is_ok();
    let build_from_source = env::var("CARGO_FEATURE_BUILD_MNN_FROM_SOURCE").is_ok();
    let static_cpp_runtime = env::var("CARGO_FEATURE_STATIC_CPP_RUNTIME").is_ok();

    let target = TargetInfo {
        os: &os,
        arch: &arch,
        env: &target_env,
        triple: &target_triple,
    };
    let features = BuildFeatures {
        coreml: coreml_enabled,
        metal: metal_enabled,
        cuda: cuda_enabled,
        opencl: opencl_enabled,
        opengl: opengl_enabled,
        vulkan: vulkan_enabled,
        mnn_dynamic,
        mnn_static,
        build_from_source,
        static_cpp_runtime,
    };
    let link_mode = select_link_mode(&target, &features)
        .unwrap_or_else(|error| panic!("Invalid MNN build configuration: {}", error));

    if matches!(link_mode, MnnLinkMode::BuildFromSource) && !build_from_source {
        let backends = features.requested_backends();
        if backends.is_empty() {
            println!(
                "cargo:warning=No compatible prebuilt MNN available for {}, building from source...",
                target_triple
            );
        } else {
            println!(
                "cargo:warning=Prebuilt MNN does not contain backend(s) {}; building MNN from source...",
                backends.join(", ")
            );
        }
    }

    let manifest_dir_path = PathBuf::from(&manifest_dir);

    // Determine MNN include dir and library dir based on link mode
    let (mnn_include_dir, mnn_lib_dir) = match &link_mode {
        MnnLinkMode::Prebuilt => {
            let asset_name = prebuilt_asset_name(&target, MNN_PREBUILT_VERSION)
                .expect("No prebuilt available (should have been caught earlier)");
            let prebuilt_dir = download_prebuilt_mnn(&manifest_dir_path, &asset_name, &os);

            let include_dir = prebuilt_dir.join("include");
            let lib_dir = prebuilt_dir.join("lib");

            if !include_dir.exists() {
                panic!(
                    "Prebuilt MNN include directory not found: {}",
                    include_dir.display()
                );
            }
            if !lib_dir.exists() {
                panic!(
                    "Prebuilt MNN lib directory not found: {}",
                    lib_dir.display()
                );
            }

            println!(
                "cargo:warning=Using prebuilt MNN {} for {}/{}",
                MNN_PREBUILT_VERSION, os, arch
            );

            (vec![include_dir], vec![lib_dir])
        }
        MnnLinkMode::BuildFromSource => {
            // Get or download MNN source code
            let mnn_source_dir = get_mnn_source(&manifest_dir_path);

            // Build MNN using cmake
            let dst = build_mnn_with_cmake(&mnn_source_dir, &target, &debug, &features);

            // Include dirs: cmake output + MNN source
            let include_dir = vec![dst.join("include"), mnn_source_dir.join("include")];
            let lib_dir = vec![dst.clone(), dst.join("lib")];
            (include_dir, lib_dir)
        }
        MnnLinkMode::Dynamic | MnnLinkMode::Static => {
            let mode_name = if mnn_dynamic {
                "mnn-dynamic"
            } else {
                "mnn-static"
            };

            // MNN_LIB_DIR is required for pre-built libraries
            let lib_dir_str = env::var("MNN_LIB_DIR").unwrap_or_else(|_| {
                panic!(
                    "MNN_LIB_DIR environment variable is required when using `{}` feature.\n\
                     Set it to the directory containing the pre-built MNN library.\n\
                     Example: MNN_LIB_DIR=/usr/local/lib cargo build --features {}",
                    mode_name, mode_name,
                )
            });
            let lib_dir = PathBuf::from(&lib_dir_str);
            if !lib_dir.exists() {
                panic!("MNN_LIB_DIR='{}' does not exist", lib_dir.display());
            }

            // MNN_INCLUDE_DIR: look for it in env, or fall back to MNN source/3rd_party
            let include_dirs = get_mnn_include_dirs(&manifest_dir_path);

            println!("cargo:rerun-if-env-changed=MNN_LIB_DIR");
            println!("cargo:rerun-if-env-changed=MNN_INCLUDE_DIR");

            println!(
                "cargo:warning=Using pre-built MNN {} library from: {}",
                if mnn_dynamic { "dynamic" } else { "static" },
                lib_dir.display()
            );

            (include_dirs, vec![lib_dir])
        }
    };

    // Build our C++ wrapper using cc (always needed)
    build_wrapper(&manifest_dir_path, &mnn_include_dir, &target, &link_mode);

    // Link libraries
    link_libraries(&mnn_lib_dir, &target, &link_mode, &features);

    // Generate Rust bindings
    bind_gen(
        &manifest_dir_path,
        &mnn_include_dir,
        &os,
        &arch,
        &target_triple,
    );
}

/// Get MNN include directories for pre-built library mode.
/// Priority:
/// 1. MNN_INCLUDE_DIR environment variable
/// 2. MNN_SOURCE_DIR/include (if MNN_SOURCE_DIR is set)
/// 3. Local 3rd_party/MNN/include
fn get_mnn_include_dirs(manifest_dir: &Path) -> Vec<PathBuf> {
    // 1. Check MNN_INCLUDE_DIR
    if let Ok(include_dir) = env::var("MNN_INCLUDE_DIR") {
        let include_path = PathBuf::from(&include_dir);
        if include_path.exists() {
            println!(
                "cargo:warning=Using MNN headers from MNN_INCLUDE_DIR: {}",
                include_path.display()
            );
            return vec![include_path];
        } else {
            panic!(
                "MNN_INCLUDE_DIR='{}' does not exist",
                include_path.display()
            );
        }
    }

    // 2. Check MNN_SOURCE_DIR
    if let Ok(mnn_dir) = env::var("MNN_SOURCE_DIR") {
        let mnn_path = PathBuf::from(&mnn_dir);
        let include_path = mnn_path.join("include");
        if include_path.exists() {
            println!(
                "cargo:warning=Using MNN headers from MNN_SOURCE_DIR: {}",
                include_path.display()
            );
            return vec![include_path];
        }
    }

    // 3. Check local 3rd_party/MNN/include
    let local_include = manifest_dir.join("3rd_party/MNN/include");
    if local_include.exists() {
        println!(
            "cargo:warning=Using MNN headers from local source: {}",
            local_include.display()
        );
        return vec![local_include];
    }

    panic!(
        "MNN headers not found. Please set one of:\n\
         - MNN_INCLUDE_DIR: path to directory containing MNN headers\n\
         - MNN_SOURCE_DIR: path to MNN source tree\n\
         Or ensure 3rd_party/MNN exists in the project root."
    );
}

/// Download and extract prebuilt MNN library from GitHub releases.
/// Returns the path to the extracted directory containing lib/ and include/.
fn download_prebuilt_mnn(manifest_dir: &Path, asset_name: &str, os: &str) -> PathBuf {
    let local_cache_dir = manifest_dir.join("3rd_party").join("prebuilt");
    let local_extract_dir = local_cache_dir.join(asset_name);

    // Keep using an existing checkout-level cache for local developer builds,
    // but never create it from build.rs: cargo publish verification forbids
    // modifying the package source directory.
    if local_extract_dir.join("lib").exists() && local_extract_dir.join("include").exists() {
        println!(
            "cargo:warning=Using cached prebuilt MNN from: {}",
            local_extract_dir.display()
        );
        // Ensure dynamic libs are removed even from cached extractions
        remove_dynamic_libs(&local_extract_dir);
        return local_extract_dir;
    }

    let cache_dir =
        PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is not set")).join("prebuilt");
    let extract_dir = cache_dir.join(asset_name);

    if extract_dir.join("lib").exists() && extract_dir.join("include").exists() {
        println!(
            "cargo:warning=Using OUT_DIR prebuilt MNN from: {}",
            extract_dir.display()
        );
        remove_dynamic_libs(&extract_dir);
        return extract_dir;
    }

    fs::create_dir_all(&cache_dir).expect("Failed to create prebuilt cache directory");

    // Determine archive extension and download URL
    let (ext, url) = if os == "windows" {
        (
            "zip",
            format!(
                "https://github.com/{}/releases/download/{}/{}.zip",
                MNN_PREBUILT_REPO, MNN_PREBUILT_VERSION, asset_name
            ),
        )
    } else {
        (
            "tar.gz",
            format!(
                "https://github.com/{}/releases/download/{}/{}.tar.gz",
                MNN_PREBUILT_REPO, MNN_PREBUILT_VERSION, asset_name
            ),
        )
    };

    let archive_path = cache_dir.join(format!("{}.{}", asset_name, ext));

    // Download if archive doesn't exist
    if !archive_path.exists() {
        println!("cargo:warning=Downloading prebuilt MNN from: {}", url);
        download_file(&url, &archive_path);
    }

    // Extract
    println!(
        "cargo:warning=Extracting prebuilt MNN to: {}",
        extract_dir.display()
    );

    if os == "windows" {
        extract_zip(&archive_path, &cache_dir);
    } else {
        extract_tar_gz(&archive_path, &cache_dir);
    }

    // Verify extraction
    if !extract_dir.join("lib").exists() {
        panic!(
            "Prebuilt MNN extraction failed: lib/ not found in {}",
            extract_dir.display()
        );
    }

    // For Windows, reorganize lib files:
    // prebuilt has MNN_static.lib -> rename to MNN.lib for static linking
    if os == "windows" {
        let lib_dir = extract_dir.join("lib");
        let static_lib = lib_dir.join("MNN_static.lib");
        let mnn_lib = lib_dir.join("MNN.lib");
        if static_lib.exists() {
            // MNN.lib from prebuilt is the import lib for DLL, we want the static one
            // Backup the import lib and replace with static lib
            let import_lib = lib_dir.join("MNN_import.lib");
            if mnn_lib.exists() {
                let _ = fs::rename(&mnn_lib, &import_lib);
            }
            fs::copy(&static_lib, &mnn_lib).expect("Failed to copy MNN_static.lib to MNN.lib");
        }
    }

    // Remove dynamic libraries to force static linking.
    // On macOS the linker prefers .dylib over .a even with `static=MNN`.
    remove_dynamic_libs(&extract_dir);

    extract_dir
}

/// Remove dynamic library files from the prebuilt lib directory to force static linking.
fn remove_dynamic_libs(extract_dir: &Path) {
    let lib_dir = extract_dir.join("lib");
    if let Ok(entries) = fs::read_dir(&lib_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.ends_with(".dylib") || name.ends_with(".so") || name.ends_with(".dll") {
                    let _ = fs::remove_file(&path);
                }
            }
        }
    }
}

/// Download a file from a URL using available system tool.
fn download_file(url: &str, dest: &Path) {
    // Try curl first (available on all modern platforms)
    let status = Command::new("curl")
        .args(["--http1.1", "-L", "-f", "-s", "-o"])
        .arg(dest.to_str().unwrap())
        .arg(url)
        .status();

    match status {
        Ok(s) if s.success() => return,
        _ => {}
    }

    // Fallback: try powershell on Windows
    if cfg!(target_os = "windows") {
        let ps_cmd = format!(
            "Invoke-WebRequest -Uri '{}' -OutFile '{}' -UseBasicParsing",
            url,
            dest.to_str().unwrap()
        );
        let status = Command::new("powershell")
            .args(["-NoProfile", "-Command", &ps_cmd])
            .status();
        match status {
            Ok(s) if s.success() => return,
            _ => {}
        }
    }

    panic!(
        "Failed to download {}. Please ensure curl is available, \
         or download manually to: {}",
        url,
        dest.display()
    );
}

/// Extract a .tar.gz archive.
fn extract_tar_gz(archive: &Path, dest_dir: &Path) {
    let status = Command::new("tar")
        .args(["xzf"])
        .arg(archive.to_str().unwrap())
        .args(["-C"])
        .arg(dest_dir.to_str().unwrap())
        .status()
        .expect("Failed to run tar");

    if !status.success() {
        panic!("Failed to extract {}", archive.display());
    }
}

/// Extract a .zip archive.
fn extract_zip(archive: &Path, dest_dir: &Path) {
    // On Windows, use powershell's Expand-Archive
    if cfg!(target_os = "windows") {
        let ps_cmd = format!(
            "Expand-Archive -Force -Path '{}' -DestinationPath '{}'",
            archive.to_str().unwrap(),
            dest_dir.to_str().unwrap()
        );
        let status = Command::new("powershell")
            .args(["-NoProfile", "-Command", &ps_cmd])
            .status()
            .expect("Failed to run powershell");
        if !status.success() {
            panic!("Failed to extract {}", archive.display());
        }
    } else {
        // Fallback: unzip command
        let status = Command::new("unzip")
            .args(["-o", "-q"])
            .arg(archive.to_str().unwrap())
            .args(["-d"])
            .arg(dest_dir.to_str().unwrap())
            .status()
            .expect("Failed to run unzip");
        if !status.success() {
            panic!("Failed to extract {}", archive.display());
        }
    }
}

/// Get MNN source code directory
/// Priority:
/// 1. Environment variable MNN_SOURCE_DIR
/// 2. Local 3rd_party/MNN directory
/// 3. Clone from GitHub
fn get_mnn_source(manifest_dir: &Path) -> PathBuf {
    // Check environment variable first
    if let Ok(mnn_dir) = env::var("MNN_SOURCE_DIR") {
        let mnn_path = PathBuf::from(mnn_dir);
        if mnn_path.exists() && mnn_path.join("CMakeLists.txt").exists() {
            println!(
                "cargo:warning=Using MNN source from MNN_SOURCE_DIR: {}",
                mnn_path.display()
            );
            return mnn_path;
        } else {
            panic!(
                "MNN_SOURCE_DIR is set but directory is invalid or missing CMakeLists.txt: {}",
                mnn_path.display()
            );
        }
    }

    // Check local 3rd_party/MNN
    let local_mnn = manifest_dir.join("3rd_party/MNN");
    if local_mnn.exists() && local_mnn.join("CMakeLists.txt").exists() {
        println!(
            "cargo:warning=Using local MNN source: {}",
            local_mnn.display()
        );
        return local_mnn;
    }

    // Clone from GitHub
    println!("cargo:warning=MNN source not found, cloning from GitHub...");
    let third_party_dir = manifest_dir.join("3rd_party");
    fs::create_dir_all(&third_party_dir).expect("Failed to create 3rd_party directory");

    let status = Command::new("git")
        .args([
            "clone",
            "--depth=1",
            "--branch=3.4.1",
            "https://github.com/alibaba/MNN.git",
            local_mnn.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to execute git clone command. Make sure git is installed.");

    if !status.success() {
        panic!("Failed to clone MNN from GitHub");
    }

    if !local_mnn.join("CMakeLists.txt").exists() {
        panic!("MNN cloned but CMakeLists.txt not found");
    }

    println!(
        "cargo:warning=Successfully cloned MNN to: {}",
        local_mnn.display()
    );
    local_mnn
}

fn build_mnn_with_cmake(
    mnn_source_dir: &Path,
    target: &TargetInfo<'_>,
    debug: &str,
    features: &BuildFeatures,
) -> PathBuf {
    let arch = target.arch;
    let os = target.os;
    let target_env = target.env;
    let mut config = cmake::Config::new(mnn_source_dir);

    config
        .define("MNN_BUILD_SHARED_LIBS", "OFF")
        .define("MNN_BUILD_TOOLS", "OFF")
        .define("MNN_BUILD_DEMO", "OFF")
        .define("MNN_BUILD_TEST", "OFF")
        .define("MNN_BUILD_BENCHMARK", "OFF")
        .define("MNN_BUILD_QUANTOOLS", "OFF")
        .define("MNN_BUILD_CONVERTER", "OFF")
        .define("MNN_PORTABLE_BUILD", "ON")
        .define("MNN_SEP_BUILD", "OFF");

    // For Windows, always use Release mode to ensure consistent CRT linking
    if os == "windows" && target_env == "msvc" {
        // Force NMake Makefiles generator on Windows to avoid MSVC detection issues
        // This is more reliable in CI/CD environments like Jenkins
        config.generator("NMake Makefiles");
        config.define("CMAKE_BUILD_TYPE", "Release");
        // Check if we're using static CRT
        if env::var("CARGO_CFG_TARGET_FEATURE").is_ok_and(|f| f.contains("crt-static")) {
            // MNN has a specific option for static CRT on Windows
            config.define("MNN_WIN_RUNTIME_MT", "ON");

            // Also set these for extra safety
            config.define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreaded");
            config.define("CMAKE_C_FLAGS_RELEASE", "/MT /O2 /Ob2 /DNDEBUG");
            config.define("CMAKE_CXX_FLAGS_RELEASE", "/MT /O2 /Ob2 /DNDEBUG");
            config.define("CMAKE_C_FLAGS", "/MT");
            config.define("CMAKE_CXX_FLAGS", "/MT");
        }
    } else {
        // For non-MSVC targets, respect the requested build profile.
        if debug == "true" {
            config.define("CMAKE_BUILD_TYPE", "Debug");
        } else {
            config.define("CMAKE_BUILD_TYPE", "Release");
        }
    }

    // Android cross-compilation
    if os == "android" {
        let ndk = env::var("ANDROID_NDK_ROOT")
            .or_else(|_| env::var("ANDROID_NDK_HOME"))
            .or_else(|_| env::var("ANDROID_NDK"))
            .or_else(|_| env::var("NDK_HOME"))
            .expect(
                "Android NDK not found. Please set one of: ANDROID_NDK_ROOT, ANDROID_NDK_HOME, ANDROID_NDK, NDK_HOME",
            );

        config
            .define(
                "CMAKE_TOOLCHAIN_FILE",
                PathBuf::from(&ndk).join("build/cmake/android.toolchain.cmake"),
            )
            .define("ANDROID_STL", "c++_static")
            .define("ANDROID_NATIVE_API_LEVEL", "android-21")
            .define("ANDROID_TOOLCHAIN", "clang")
            .define("MNN_BUILD_FOR_ANDROID_COMMAND", "ON")
            .define("MNN_USE_SSE", "OFF");

        match arch {
            "arm" => {
                config.define("ANDROID_ABI", "armeabi-v7a");
            }
            "aarch64" => {
                config.define("ANDROID_ABI", "arm64-v8a");
            }
            "x86" => {
                config.define("ANDROID_ABI", "x86");
            }
            "x86_64" => {
                config.define("ANDROID_ABI", "x86_64");
            }
            _ => {}
        }
    }

    // iOS cross-compilation
    if os == "ios" {
        let rust_target = env::var("TARGET").unwrap_or_default();
        let is_simulator = rust_target.contains("-sim") || arch == "x86_64";

        config
            .define("CMAKE_SYSTEM_NAME", "iOS")
            .define("MNN_BUILD_FOR_IOS", "ON")
            .define("CMAKE_OSX_DEPLOYMENT_TARGET", "13.0");

        if arch == "aarch64" {
            config.define("CMAKE_OSX_ARCHITECTURES", "arm64");
        } else if arch == "x86_64" {
            config.define("CMAKE_OSX_ARCHITECTURES", "x86_64");
        }

        // Critical: set the correct SDK for simulator vs device
        if is_simulator {
            config.define("CMAKE_OSX_SYSROOT", "iphonesimulator");
        } else {
            config.define("CMAKE_OSX_SYSROOT", "iphoneos");
        }

        // MNN's CMakeLists.txt only sets CMAKE_SYSTEM_PROCESSOR from
        // CMAKE_OSX_ARCHITECTURES when CMAKE_SYSTEM_NAME == "Darwin",
        // but for iOS it's "iOS". Without this, ARM assembly sources
        // (NEON, AArch64) are not compiled, causing undefined symbols.
        if arch == "aarch64" {
            config.define("CMAKE_SYSTEM_PROCESSOR", "arm64");
            config.define("ARCHS", "arm64");
        } else if arch == "x86_64" {
            config.define("CMAKE_SYSTEM_PROCESSOR", "x86_64");
        }
    }

    // SIMD optimizations
    // Only enable SSE for x86_64, not for 32-bit x86 (i686)
    // because i686 target doesn't have guaranteed SSE support
    if arch == "x86_64" && os != "android" && os != "ios" {
        config.define("MNN_USE_SSE", "ON");
    } else {
        // For all other architectures (including 32-bit x86/i686), disable SSE/AVX
        // This prevents compilation errors with SIMD intrinsics on incompatible targets
        config.define("MNN_USE_SSE", "OFF");
        config.define("MNN_USE_AVX", "OFF");
        config.define("MNN_USE_AVX2", "OFF");
        config.define("MNN_USE_AVX512", "OFF");
    }

    // CoreML (macOS/iOS only)
    if features.coreml && matches!(os, "macos" | "ios") {
        config.define("MNN_COREML", "ON");
    }

    // Metal GPU (macOS/iOS only)
    if features.metal && matches!(os, "macos" | "ios") {
        config.define("MNN_METAL", "ON");
    }

    // CUDA GPU (Linux/Windows)
    if features.cuda && matches!(os, "linux" | "windows") {
        config.define("MNN_CUDA", "ON");
    }

    // OpenCL GPU (cross-platform)
    if features.opencl {
        config.define("MNN_OPENCL", "ON");
    }

    // OpenGL GPU (Android/Linux)
    if features.opengl && matches!(os, "android" | "linux") {
        config.define("MNN_OPENGL", "ON");
    }

    // Vulkan GPU (cross-platform)
    if features.vulkan {
        config.define("MNN_VULKAN", "ON");
    }

    println!("cargo:rerun-if-changed=MNN/CMakeLists.txt");

    let dst = config.build();

    if let Some(plan) = cuda_side_library_plan(os, features.cuda, MnnLinkMode::BuildFromSource) {
        let source = dst.join(
            plan.build_relative_path
                .expect("source-built CUDA library must have a build path"),
        );
        let destination = dst.join(
            plan.install_relative_path
                .expect("source-built CUDA library must have an install path"),
        );
        if !source.exists() {
            panic!(
                "MNN CUDA side library was not produced at {}",
                source.display()
            );
        }
        fs::create_dir_all(
            destination
                .parent()
                .expect("CUDA library install path must have a parent"),
        )
        .expect("Failed to create MNN CUDA library install directory");
        fs::copy(&source, &destination).unwrap_or_else(|error| {
            panic!(
                "Failed to install MNN CUDA side library from {} to {}: {}",
                source.display(),
                destination.display(),
                error
            )
        });
    }

    dst
}

fn build_wrapper(
    manifest_dir: &Path,
    mnn_include_dirs: &[PathBuf],
    target: &TargetInfo<'_>,
    link_mode: &MnnLinkMode,
) {
    let wrapper_file = manifest_dir.join("cpp/src/mnn_wrapper.cpp");

    println!("cargo:rerun-if-changed=cpp/src/mnn_wrapper.cpp");
    println!("cargo:rerun-if-changed=cpp/include/mnn_wrapper.h");

    let mut build = cc::Build::new();

    build
        .cpp(true)
        .cpp_link_stdlib(None::<&str>)
        .file(&wrapper_file)
        .include(manifest_dir.join("cpp/include"));

    for inc in mnn_include_dirs {
        build.include(inc);
    }

    // Platform-specific C++ flags
    if uses_msvc_flags(target)
        .unwrap_or_else(|error| panic!("Invalid C++ target configuration: {}", error))
    {
        build.flag("/std:c++14").flag("/EHsc").flag("/W3");
        // Match CRT with prebuilt MNN: prebuilt uses /MT (static CRT)
        if matches!(link_mode, MnnLinkMode::Prebuilt) {
            build.static_crt(true);
        }
    } else {
        build.flag("-std=c++14").flag("-fvisibility=hidden");
    }

    build.compile("mnn_wrapper");
}

fn link_libraries(
    lib_dirs: &[PathBuf],
    target: &TargetInfo<'_>,
    link_mode: &MnnLinkMode,
    features: &BuildFeatures,
) {
    let os = target.os;

    emit_static_cpp_runtime_search_paths(target, features);

    // Add library search paths
    for dir in lib_dirs {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }

    // Link MNN library based on mode
    match link_mode {
        MnnLinkMode::Dynamic => {
            println!("cargo:rustc-link-lib=dylib=MNN");
        }
        MnnLinkMode::Static | MnnLinkMode::BuildFromSource | MnnLinkMode::Prebuilt => {
            if should_link_mnn_whole_archive(*link_mode, features) {
                println!("cargo:rustc-link-lib=static:+whole-archive=MNN");
            } else {
                println!("cargo:rustc-link-lib=static=MNN");
            }
        }
    }

    // Link the C++ runtime after MNN so GNU static linking can resolve MNN's symbols.
    for library in cpp_runtime_libraries(target, features.static_cpp_runtime) {
        match library.kind {
            NativeLinkKind::Dynamic => println!("cargo:rustc-link-lib=dylib={}", library.name),
            NativeLinkKind::Static => println!("cargo:rustc-link-lib=static={}", library.name),
        }
    }

    // Other platform-specific system libraries
    match os {
        "linux" => {
            println!("cargo:rustc-link-lib=m");
            println!("cargo:rustc-link-lib=pthread");
        }
        "android" => {
            println!("cargo:rustc-link-lib=log");
        }
        _ => {}
    }

    // Prebuilt MNN for macOS/iOS includes Metal backend, so always link Apple frameworks
    if matches!(link_mode, MnnLinkMode::Prebuilt) && matches!(os, "macos" | "ios") {
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        println!("cargo:rustc-link-lib=objc");
    }

    // CoreML frameworks
    if features.coreml && matches!(os, "macos" | "ios") {
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    }

    // Metal frameworks
    if features.metal && matches!(os, "macos" | "ios") {
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    }

    // CUDA libraries
    if features.cuda && matches!(os, "linux" | "windows") {
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        if let Some(plan) = cuda_side_library_plan(os, features.cuda, *link_mode) {
            println!("cargo:rustc-link-lib=dylib={}", plan.link_name);
        }
    }

    // OpenCL library
    if features.opencl {
        if os == "macos" {
            println!("cargo:rustc-link-lib=framework=OpenCL");
        } else {
            println!("cargo:rustc-link-lib=OpenCL");
        }
    }

    // OpenGL libraries
    if features.opengl && matches!(os, "android" | "linux") {
        if os == "android" {
            println!("cargo:rustc-link-lib=GLESv3");
            println!("cargo:rustc-link-lib=EGL");
        } else {
            println!("cargo:rustc-link-lib=GL");
        }
    }

    // Vulkan library
    if features.vulkan {
        println!("cargo:rustc-link-lib=vulkan");
    }
}

fn emit_static_cpp_runtime_search_paths(target: &TargetInfo<'_>, features: &BuildFeatures) {
    if target.os != "windows" || target.env != "gnu" || !features.static_cpp_runtime {
        return;
    }

    let compiler = cc::Build::new().cpp(true).get_compiler();
    let mut emitted = HashSet::new();

    for library in cpp_runtime_libraries(target, true) {
        if library.kind != NativeLinkKind::Static {
            continue;
        }

        let archive_name = format!("lib{}.a", library.name);
        let output = compiler
            .to_command()
            .arg(format!("-print-file-name={archive_name}"))
            .output()
            .unwrap_or_else(|error| {
                panic!("Failed to query MinGW C++ compiler for {archive_name}: {error}")
            });

        if !output.status.success() {
            panic!(
                "MinGW C++ compiler failed to locate {archive_name}: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }

        let archive = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
        if !archive.is_absolute() || !archive.is_file() {
            panic!(
                "feature `static-cpp-runtime` requires {archive_name}, but the target C++ compiler could not locate it"
            );
        }

        let directory = archive
            .parent()
            .expect("MinGW runtime archive should have a parent directory")
            .to_path_buf();
        if emitted.insert(directory.clone()) {
            println!("cargo:rustc-link-search=native={}", directory.display());
        }
    }
}

fn bind_gen(
    manifest_dir: &Path,
    mnn_include_dirs: &[PathBuf],
    os: &str,
    arch: &str,
    target_triple: &str,
) {
    let header_path = manifest_dir.join("cpp/include/mnn_wrapper.h");

    let mut builder = bindgen::Builder::default()
        .header(header_path.to_string_lossy())
        .allowlist_function("mnnr_.*")
        .allowlist_type("MNN.*")
        .allowlist_type("MNNR.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .layout_tests(false);

    for inc in mnn_include_dirs {
        builder = builder.clang_arg(format!("-I{}", inc.display()));
    }

    if os == "linux" {
        builder = add_linux_system_include_args(builder);
    }

    if os == "windows" && target_triple.contains("-gnu") {
        builder = builder.clang_arg(format!("--target={}", target_triple));
    }

    // Android-specific clang target and sysroot
    if os == "android" {
        let ndk = env::var("ANDROID_NDK_ROOT")
            .or_else(|_| env::var("ANDROID_NDK_HOME"))
            .or_else(|_| env::var("ANDROID_NDK"))
            .or_else(|_| env::var("NDK_HOME"))
            .unwrap_or_default();

        let api_level = "21";
        let target = match arch {
            "aarch64" => "aarch64-linux-android",
            "arm" => "armv7-linux-androideabi",
            "x86_64" => "x86_64-linux-android",
            "x86" => "i686-linux-android",
            _ => "aarch64-linux-android",
        };
        builder = builder.clang_arg(format!("--target={}{}", target, api_level));

        // Point bindgen to the NDK sysroot so it doesn't pick up host headers
        if !ndk.is_empty() {
            let host_tag = if cfg!(target_os = "macos") {
                "darwin-x86_64"
            } else {
                "linux-x86_64"
            };
            let sysroot = PathBuf::from(&ndk)
                .join("toolchains/llvm/prebuilt")
                .join(host_tag)
                .join("sysroot");
            if sysroot.exists() {
                builder = builder.clang_arg(format!("--sysroot={}", sysroot.display()));
            }
        }
    }

    // iOS-specific: remap target triple for clang/bindgen compatibility
    if os == "ios" {
        let rust_target = env::var("TARGET").unwrap_or_default();
        let clang_target = if rust_target == "aarch64-apple-ios-sim" {
            "arm64-apple-ios13.0-simulator".to_string()
        } else if rust_target == "aarch64-apple-ios" {
            "arm64-apple-ios13.0".to_string()
        } else if rust_target == "x86_64-apple-ios" {
            "x86_64-apple-ios13.0-simulator".to_string()
        } else {
            rust_target
        };
        builder = builder.clang_arg(format!("--target={}", clang_target));
    }

    let bindings = builder.generate().expect("Unable to generate bindings");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    fs::write(out_path.join("mnn_bindings.rs"), bindings.to_string())
        .expect("Couldn't write bindings!");
}

fn add_linux_system_include_args(mut builder: bindgen::Builder) -> bindgen::Builder {
    let mut include_dirs = Vec::new();
    let mut seen = HashSet::new();

    let compiler = cc::Build::new().get_compiler();
    let compiler_path = compiler.path();

    if let Some(include_dir) = command_path_output(compiler_path, &["-print-file-name=include"]) {
        push_unique_path(&mut include_dirs, &mut seen, PathBuf::from(include_dir));
    }

    let sysroot = command_path_output(compiler_path, &["-print-sysroot"])
        .filter(|value| !value.is_empty() && value != "/");

    let target_include = command_path_output(compiler_path, &["-dumpmachine"])
        .map(PathBuf::from)
        .or_else(|| env::var("TARGET").ok().map(PathBuf::from));

    if let Some(sysroot) = sysroot.as_ref() {
        let sysroot_path = PathBuf::from(sysroot);
        push_unique_path(
            &mut include_dirs,
            &mut seen,
            sysroot_path.join("usr/local/include"),
        );
        if let Some(target) = target_include.as_ref() {
            push_unique_path(
                &mut include_dirs,
                &mut seen,
                sysroot_path.join("usr/include").join(target),
            );
        }
        push_unique_path(
            &mut include_dirs,
            &mut seen,
            sysroot_path.join("usr/include"),
        );
    }

    push_unique_path(
        &mut include_dirs,
        &mut seen,
        PathBuf::from("/usr/local/include"),
    );
    if let Some(target) = target_include.as_ref() {
        push_unique_path(
            &mut include_dirs,
            &mut seen,
            PathBuf::from("/usr/include").join(target),
        );
    }
    push_unique_path(&mut include_dirs, &mut seen, PathBuf::from("/usr/include"));

    for dir in include_dirs {
        println!(
            "cargo:warning=Adding Linux system include for bindgen: {}",
            dir.display()
        );
        builder = builder.clang_arg(format!("-isystem{}", dir.display()));
    }

    builder
}

fn command_path_output(program: &std::path::Path, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }

    let value = String::from_utf8(output.stdout).ok()?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn push_unique_path(paths: &mut Vec<PathBuf>, seen: &mut HashSet<PathBuf>, path: PathBuf) {
    if path.exists() && seen.insert(path.clone()) {
        paths.push(path);
    }
}
