use std::path::PathBuf;
use std::process::Command;
use std::{env, fs};

fn main() {
    // 在 docs.rs 构建环境中，跳过所有 C++ 编译
    if env::var("DOCS_RS").is_ok() || env::var("CARGO_FEATURE_DOCSRS").is_ok() {
        println!("cargo:warning=Building for docs.rs, skipping C++ compilation");
        return;
    }

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let debug = env::var("DEBUG").unwrap();

    // Feature flags
    let coreml_enabled = env::var("CARGO_FEATURE_COREML").is_ok();
    let metal_enabled = env::var("CARGO_FEATURE_METAL").is_ok();
    let cuda_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();
    let opencl_enabled = env::var("CARGO_FEATURE_OPENCL").is_ok();
    let opengl_enabled = env::var("CARGO_FEATURE_OPENGL").is_ok();
    let vulkan_enabled = env::var("CARGO_FEATURE_VULKAN").is_ok();

    let manifest_dir_path = PathBuf::from(&manifest_dir);

    // Get or download MNN source code
    let mnn_source_dir = get_mnn_source(&manifest_dir_path);

    // Build MNN using cmake
    let dst = build_mnn_with_cmake(
        &mnn_source_dir,
        &arch,
        &os,
        &debug,
        coreml_enabled,
        metal_enabled,
        cuda_enabled,
        opencl_enabled,
        opengl_enabled,
        vulkan_enabled,
    );

    // Build our C++ wrapper using cc
    build_wrapper(&manifest_dir_path, &mnn_source_dir, &dst, &os);

    // Link libraries
    link_libraries(
        &dst,
        &os,
        coreml_enabled,
        metal_enabled,
        cuda_enabled,
        opencl_enabled,
        opengl_enabled,
        vulkan_enabled,
    );

    // Generate Rust bindings
    bind_gen(&manifest_dir_path, &mnn_source_dir, &dst, &os, &arch);
}

fn is_valid_mnn_dir(p: &PathBuf) -> bool {
    p.exists() && p.is_dir() && p.join("CMakeLists.txt").exists()
}

fn dir_exists_and_nonempty(p: &PathBuf) -> bool {
    if !p.exists() || !p.is_dir() {
        return false;
    }
    match fs::read_dir(p) {
        Ok(mut it) => it.next().is_some(),
        Err(_) => false,
    }
}


/// Get MNN source code directory
/// Priority:
/// 1. Environment variable MNN_SOURCE_DIR
/// 2. Local 3rd_party/MNN directory
/// 3. Clone from GitHub
fn get_mnn_source(manifest_dir: &PathBuf) -> PathBuf {
    // 1) Check environment variable first
    if let Ok(mnn_dir) = env::var("MNN_SOURCE_DIR") {
        let mnn_path = PathBuf::from(mnn_dir);
        if is_valid_mnn_dir(&mnn_path) {
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

    // 2) local third_party/MNN
    let local_mnn = manifest_dir.join("3rd_party/MNN");
    if is_valid_mnn_dir(&local_mnn) {
        println!("cargo:warning=Using local MNN source: {}", local_mnn.display());
        return local_mnn;
    }

    // 3) If local_mnn exists but invalid, remove it (fix your current error)
    //    This is the key change: make clone step idempotent.
    if dir_exists_and_nonempty(&local_mnn) {
        println!(
            "cargo:warning=Found existing but invalid MNN dir (missing CMakeLists.txt). Removing: {}",
            local_mnn.display()
        );
        fs::remove_dir_all(&local_mnn)
            .expect("Failed to remove existing invalid 3rd_party/MNN directory");
    } else if local_mnn.exists() {
        // exists but empty or weird - remove as well
        let _ = fs::remove_dir_all(&local_mnn);
    }

    // 4) clone
    if env::var("MNN_NO_NETWORK").is_ok() {
        panic!(
            "MNN source not found locally and MNN_NO_NETWORK is set. \
            Provide MNN_SOURCE_DIR or add 3rd_party/MNN with CMakeLists.txt."
        );
    }

    println!("cargo:warning=MNN source not found, cloning from GitHub...");
    let third_party_dir = manifest_dir.join("3rd_party");
    fs::create_dir_all(&third_party_dir).expect("Failed to create 3rd_party directory");

    let status = Command::new("git")
        .args(&[
            "clone",
            "--depth=1",
            "https://github.com/alibaba/MNN.git",
            local_mnn.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to execute git clone command. Make sure git is installed.");

    if !status.success() {
        panic!(
            "Failed to clone MNN from GitHub. If you want to avoid network cloning, set MNN_SOURCE_DIR to a local MNN checkout."
        );
    }

    let status = Command::new("git")
    .current_dir(&local_mnn)
    .args(&["checkout", "9bd8302"])
    .status()
    .expect("Failed to checkout MNN commit 9bd8302");

    if !status.success() {
        panic!("Failed to checkout MNN commit 9bd8302");
    }


    if !is_valid_mnn_dir(&local_mnn) {
        panic!("MNN cloned but CMakeLists.txt not found");
    }

    println!("cargo:warning=Successfully cloned MNN to: {}", local_mnn.display());
    local_mnn
}


fn build_mnn_with_cmake(
    mnn_source_dir: &PathBuf,
    arch: &str,
    os: &str,
    debug: &str,
    coreml_enabled: bool,
    metal_enabled: bool,
    cuda_enabled: bool,
    opencl_enabled: bool,
    opengl_enabled: bool,
    vulkan_enabled: bool,
) -> PathBuf {
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
    if os == "windows" {
        config.define("CMAKE_BUILD_TYPE", "Release");
        // Check if we're using static CRT
        if env::var("CARGO_CFG_TARGET_FEATURE").map_or(false, |f| f.contains("crt-static")) {
            // MNN has a specific option for static CRT on Windows
            config.define("MNN_WIN_RUNTIME_MT", "ON");

            // Also set these for extra safety
            config.define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreaded");
            config.define("CMAKE_C_FLAGS_RELEASE", "/MT /O2 /Ob2 /DNDEBUG");
            config.define("CMAKE_CXX_FLAGS_RELEASE", "/MT /O2 /Ob2 /DNDEBUG");
            config.define("CMAKE_C_FLAGS", "/MT");
            config.define("CMAKE_CXX_FLAGS", "/MT");
        }
        // For 32-bit Windows, ensure proper configuration
        if arch == "x86" {
            config.define("CMAKE_GENERATOR_PLATFORM", "Win32");
        }
    } else {
        // For non-Windows platforms, respect debug flag
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
        config
            .define("CMAKE_SYSTEM_NAME", "iOS")
            .define("MNN_BUILD_FOR_IOS", "ON");

        if arch == "aarch64" {
            config.define("CMAKE_OSX_ARCHITECTURES", "arm64");
        } else if arch == "x86_64" {
            // Simulator
            config.define("CMAKE_OSX_ARCHITECTURES", "x86_64");
        }
    }

    // SIMD optimizations
    if matches!(arch, "x86" | "x86_64") && os != "android" {
        config.define("MNN_USE_SSE", "ON");
    }

    // CoreML (macOS/iOS only)
    if coreml_enabled && matches!(os, "macos" | "ios") {
        config.define("MNN_COREML", "ON");
    }

    // Metal GPU (macOS/iOS only)
    if metal_enabled && matches!(os, "macos" | "ios") {
        config.define("MNN_METAL", "ON");
    }

    // CUDA GPU (Linux/Windows)
    if cuda_enabled && matches!(os, "linux" | "windows") {
        config.define("MNN_CUDA", "ON");
    }

    // OpenCL GPU (cross-platform)
    if opencl_enabled {
        config.define("MNN_OPENCL", "ON");
    }

    // OpenGL GPU (Android/Linux)
    if opengl_enabled && matches!(os, "android" | "linux") {
        config.define("MNN_OPENGL", "ON");
    }

    // Vulkan GPU (cross-platform)
    if vulkan_enabled {
        config.define("MNN_VULKAN", "ON");
    }

    println!("cargo:rerun-if-env-changed=MNN_SOURCE_DIR");
    println!("cargo:rerun-if-env-changed=MNN_NO_NETWORK");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=3rd_party/MNN/CMakeLists.txt");

    config.build()
}

fn build_wrapper(manifest_dir: &PathBuf, mnn_source_dir: &PathBuf, mnn_dst: &PathBuf, os: &str) {
    let wrapper_file = manifest_dir.join("cpp/src/mnn_wrapper.cpp");

    println!("cargo:rerun-if-changed=cpp/src/mnn_wrapper.cpp");
    println!("cargo:rerun-if-changed=cpp/include/mnn_wrapper.h");

    let mut build = cc::Build::new();

    build
        .cpp(true)
        .file(&wrapper_file)
        .include(mnn_dst.join("include"))
        .include(mnn_source_dir.join("include"))
        .include(manifest_dir.join("cpp/include"));

    // Platform-specific C++ flags
    if os == "windows" {
        build.flag("/std:c++14").flag("/EHsc").flag("/W3");
    } else {
        build.flag("-std=c++14").flag("-fvisibility=hidden");
    }

    build.compile("mnn_wrapper");
}

fn link_libraries(
    mnn_dst: &PathBuf,
    os: &str,
    coreml_enabled: bool,
    metal_enabled: bool,
    cuda_enabled: bool,
    opencl_enabled: bool,
    opengl_enabled: bool,
    vulkan_enabled: bool,
) {
    // Add library search paths
    println!("cargo:rustc-link-search=native={}", mnn_dst.display());
    println!(
        "cargo:rustc-link-search=native={}",
        mnn_dst.join("lib").display()
    );

    // Link MNN static library
    println!("cargo:rustc-link-lib=static=MNN");

    // Platform-specific C++ runtime
    match os {
        "macos" | "ios" => {
            println!("cargo:rustc-link-lib=c++");
        }
        "linux" => {
            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-link-lib=m");
            println!("cargo:rustc-link-lib=pthread");
        }
        "android" => {
            println!("cargo:rustc-link-lib=c++_static");
            println!("cargo:rustc-link-lib=log");
        }
        "windows" => {
            // MSVC runtime is linked automatically when using matching CRT settings
        }
        _ => {}
    }

    // CoreML frameworks
    if coreml_enabled && matches!(os, "macos" | "ios") {
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    }

    // Metal frameworks
    if metal_enabled && matches!(os, "macos" | "ios") {
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    }

    // CUDA libraries
    if cuda_enabled && matches!(os, "linux" | "windows") {
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cudnn");
    }

    // OpenCL library
    if opencl_enabled {
        if os == "macos" {
            println!("cargo:rustc-link-lib=framework=OpenCL");
        } else {
            println!("cargo:rustc-link-lib=OpenCL");
        }
    }

    // OpenGL libraries
    if opengl_enabled && matches!(os, "android" | "linux") {
        if os == "android" {
            println!("cargo:rustc-link-lib=GLESv3");
            println!("cargo:rustc-link-lib=EGL");
        } else {
            println!("cargo:rustc-link-lib=GL");
        }
    }

    // Vulkan library
    if vulkan_enabled {
        println!("cargo:rustc-link-lib=vulkan");
    }
}

fn bind_gen(
    manifest_dir: &PathBuf,
    mnn_source_dir: &PathBuf,
    mnn_dst: &PathBuf,
    os: &str,
    arch: &str,
) {
    let header_path = manifest_dir.join("cpp/include/mnn_wrapper.h");

    let mut builder = bindgen::Builder::default()
        .header(header_path.to_string_lossy())
        .allowlist_function("mnnr_.*")
        .allowlist_type("MNN.*")
        .allowlist_type("MNNR.*")
        .clang_arg(format!("-I{}", mnn_dst.join("include").display()))
        .clang_arg(format!("-I{}", mnn_source_dir.join("include").display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .layout_tests(false);

    // Android-specific clang target
    if os == "android" {
        let target = match arch {
            "aarch64" => "aarch64-linux-android",
            "arm" => "armv7-linux-androideabi",
            "x86_64" => "x86_64-linux-android",
            "x86" => "i686-linux-android",
            _ => "aarch64-linux-android",
        };
        builder = builder.clang_arg(format!("--target={}", target));
    }

    let bindings = builder.generate().expect("Unable to generate bindings");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    fs::write(out_path.join("mnn_bindings.rs"), bindings.to_string())
        .expect("Couldn't write bindings!");
}
