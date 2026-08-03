//! Pure build-planning helpers shared by `build.rs` and its tests.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MnnLinkMode {
    Prebuilt,
    BuildFromSource,
    Dynamic,
    Static,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaSideLibraryPlan {
    pub link_name: &'static str,
    pub build_relative_path: Option<&'static str>,
    pub install_relative_path: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BuildFeatures {
    pub coreml: bool,
    pub metal: bool,
    pub cuda: bool,
    pub opencl: bool,
    pub opengl: bool,
    pub vulkan: bool,
    pub mnn_dynamic: bool,
    pub mnn_static: bool,
    pub build_from_source: bool,
    pub static_cpp_runtime: bool,
}

impl BuildFeatures {
    pub fn requested_backends(self) -> Vec<&'static str> {
        [
            (self.coreml, "coreml"),
            (self.metal, "metal"),
            (self.cuda, "cuda"),
            (self.opencl, "opencl"),
            (self.opengl, "opengl"),
            (self.vulkan, "vulkan"),
        ]
        .into_iter()
        .filter_map(|(enabled, name)| enabled.then_some(name))
        .collect()
    }

    fn requests_backend(self) -> bool {
        !self.requested_backends().is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetInfo<'a> {
    pub os: &'a str,
    pub arch: &'a str,
    pub env: &'a str,
    pub triple: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildConfigError<'a> {
    ConflictingLinkModes,
    UnsupportedWindowsEnvironment(&'a str),
    UnsupportedBackendForTarget {
        backend: &'static str,
        target: &'a str,
    },
}

impl fmt::Display for BuildConfigError<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConflictingLinkModes => formatter.write_str(
                "features `mnn-dynamic` and `mnn-static` are mutually exclusive",
            ),
            Self::UnsupportedWindowsEnvironment(environment) => write!(
                formatter,
                "unsupported Windows target environment `{environment}`; use `msvc` or `gnu`"
            ),
            Self::UnsupportedBackendForTarget { backend, target } => write!(
                formatter,
                "backend `{backend}` cannot be built from source for target `{target}`; provide a compatible MNN library with `mnn-dynamic` or `mnn-static`"
            ),
        }
    }
}

impl std::error::Error for BuildConfigError<'_> {}

pub fn prebuilt_asset_name(target: &TargetInfo<'_>, version: &str) -> Option<String> {
    let suffix = match (target.os, target.arch) {
        ("linux", "x86_64") => "linux-x86_64",
        ("linux", "aarch64") => "linux-aarch64",
        ("windows", "x86_64") if target.env == "msvc" => "windows-x86_64",
        ("windows", "x86") if target.env == "msvc" => "windows-i686",
        ("windows", "aarch64") if target.env == "msvc" => "windows-aarch64",
        ("macos", _) => "macos-universal",
        ("ios", "aarch64") if target.triple.contains("-sim") => "ios-arm64-sim",
        ("ios", "aarch64") => "ios-arm64",
        ("android", "aarch64") => "android-arm64-v8a",
        ("android", "arm") => "android-armeabi-v7a",
        _ => return None,
    };

    Some(format!("mnn-{version}-{suffix}"))
}

fn prebuilt_supports_requested_backends(target: &TargetInfo<'_>, features: &BuildFeatures) -> bool {
    if !features.requests_backend() {
        return true;
    }

    matches!(target.os, "macos" | "ios")
        && features.metal
        && !features.coreml
        && !features.cuda
        && !features.opencl
        && !features.opengl
        && !features.vulkan
}

fn validate_windows_environment<'a>(target: &TargetInfo<'a>) -> Result<(), BuildConfigError<'a>> {
    if target.os == "windows" && !matches!(target.env, "msvc" | "gnu") {
        return Err(BuildConfigError::UnsupportedWindowsEnvironment(target.env));
    }
    Ok(())
}

fn validate_source_backend_support<'a>(
    target: &TargetInfo<'a>,
    features: &BuildFeatures,
) -> Result<(), BuildConfigError<'a>> {
    // NVIDIA's Windows CUDA toolchain requires the MSVC ABI. A user-supplied
    // MinGW-compatible MNN library can still be selected explicitly.
    if target.os == "windows" && target.env == "gnu" && features.cuda {
        return Err(BuildConfigError::UnsupportedBackendForTarget {
            backend: "cuda",
            target: target.triple,
        });
    }
    Ok(())
}

pub fn select_link_mode<'a>(
    target: &TargetInfo<'a>,
    features: &BuildFeatures,
) -> Result<MnnLinkMode, BuildConfigError<'a>> {
    validate_windows_environment(target)?;

    if features.mnn_dynamic && features.mnn_static {
        return Err(BuildConfigError::ConflictingLinkModes);
    }
    if features.mnn_dynamic {
        return Ok(MnnLinkMode::Dynamic);
    }
    if features.mnn_static {
        return Ok(MnnLinkMode::Static);
    }

    let has_prebuilt = prebuilt_asset_name(target, "unused").is_some();
    let needs_source = features.build_from_source
        || !has_prebuilt
        || !prebuilt_supports_requested_backends(target, features);

    if needs_source {
        validate_source_backend_support(target, features)?;
        Ok(MnnLinkMode::BuildFromSource)
    } else {
        Ok(MnnLinkMode::Prebuilt)
    }
}

pub fn uses_msvc_flags<'a>(target: &TargetInfo<'a>) -> Result<bool, BuildConfigError<'a>> {
    validate_windows_environment(target)?;
    Ok(target.os == "windows" && target.env == "msvc")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeLinkKind {
    Dynamic,
    Static,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeLibrary {
    pub name: &'static str,
    pub kind: NativeLinkKind,
}

const WINDOWS_GNU_STATIC_CPP_RUNTIME: &[NativeLibrary] = &[
    NativeLibrary {
        name: "stdc++",
        kind: NativeLinkKind::Static,
    },
    NativeLibrary {
        name: "gcc_eh",
        kind: NativeLinkKind::Static,
    },
    NativeLibrary {
        name: "gcc",
        kind: NativeLinkKind::Static,
    },
    NativeLibrary {
        name: "winpthread",
        kind: NativeLinkKind::Static,
    },
];

const WINDOWS_GNU_DYNAMIC_CPP_RUNTIME: &[NativeLibrary] = &[NativeLibrary {
    name: "stdc++",
    kind: NativeLinkKind::Dynamic,
}];

const LIBCXX_RUNTIME: &[NativeLibrary] = &[NativeLibrary {
    name: "c++",
    kind: NativeLinkKind::Dynamic,
}];

const LIBSTDCXX_RUNTIME: &[NativeLibrary] = &[NativeLibrary {
    name: "stdc++",
    kind: NativeLinkKind::Dynamic,
}];

const ANDROID_CPP_RUNTIME: &[NativeLibrary] = &[NativeLibrary {
    name: "c++_static",
    kind: NativeLinkKind::Static,
}];

pub fn cpp_runtime_libraries(
    target: &TargetInfo<'_>,
    static_cpp_runtime: bool,
) -> &'static [NativeLibrary] {
    match (target.os, target.env) {
        ("windows", "gnu") if static_cpp_runtime => WINDOWS_GNU_STATIC_CPP_RUNTIME,
        ("windows", "gnu") => WINDOWS_GNU_DYNAMIC_CPP_RUNTIME,
        ("macos" | "ios", _) => LIBCXX_RUNTIME,
        ("linux", _) => LIBSTDCXX_RUNTIME,
        ("android", _) => ANDROID_CPP_RUNTIME,
        _ => &[],
    }
}

pub fn should_link_mnn_whole_archive(link_mode: MnnLinkMode, features: &BuildFeatures) -> bool {
    matches!(
        link_mode,
        MnnLinkMode::BuildFromSource | MnnLinkMode::Static
    ) && features.requests_backend()
}

pub fn cuda_side_library_plan(
    os: &str,
    cuda_enabled: bool,
    link_mode: MnnLinkMode,
) -> Option<CudaSideLibraryPlan> {
    if os != "linux" || !cuda_enabled {
        return None;
    }

    let source_paths = matches!(link_mode, MnnLinkMode::BuildFromSource).then_some((
        "build/source/backend/cuda/libMNN_Cuda_Main.so",
        "lib/libMNN_Cuda_Main.so",
    ));

    Some(CudaSideLibraryPlan {
        link_name: "MNN_Cuda_Main",
        build_relative_path: source_paths.map(|paths| paths.0),
        install_relative_path: source_paths.map(|paths| paths.1),
    })
}
