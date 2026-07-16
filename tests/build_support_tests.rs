#[path = "../build_support.rs"]
mod build_support;

use build_support::{
    prebuilt_asset_name, select_link_mode, should_link_mnn_whole_archive, uses_msvc_flags,
    BuildConfigError, BuildFeatures, MnnLinkMode, TargetInfo,
};

fn target<'a>(os: &'a str, arch: &'a str, env: &'a str, triple: &'a str) -> TargetInfo<'a> {
    TargetInfo {
        os,
        arch,
        env,
        triple,
    }
}

#[test]
fn cpu_only_linux_uses_the_prebuilt_library() {
    let target = target("linux", "x86_64", "gnu", "x86_64-unknown-linux-gnu");

    assert_eq!(
        select_link_mode(&target, &BuildFeatures::default()),
        Ok(MnnLinkMode::Prebuilt)
    );
    assert_eq!(
        prebuilt_asset_name(&target, "dev").as_deref(),
        Some("mnn-dev-linux-x86_64")
    );
}

#[test]
fn cuda_and_vulkan_force_a_source_build_when_prebuilt_has_no_gpu_backend() {
    let target = target("linux", "x86_64", "gnu", "x86_64-unknown-linux-gnu");

    for features in [
        BuildFeatures {
            cuda: true,
            ..BuildFeatures::default()
        },
        BuildFeatures {
            vulkan: true,
            ..BuildFeatures::default()
        },
    ] {
        assert_eq!(
            select_link_mode(&target, &features),
            Ok(MnnLinkMode::BuildFromSource)
        );
    }
}

#[test]
fn apple_metal_uses_the_metal_enabled_prebuilt() {
    let target = target("macos", "aarch64", "", "aarch64-apple-darwin");
    let features = BuildFeatures {
        metal: true,
        ..BuildFeatures::default()
    };

    assert_eq!(
        select_link_mode(&target, &features),
        Ok(MnnLinkMode::Prebuilt)
    );
}

#[test]
fn non_metal_apple_gpu_backend_forces_a_source_build() {
    let target = target("macos", "aarch64", "", "aarch64-apple-darwin");
    let features = BuildFeatures {
        opencl: true,
        ..BuildFeatures::default()
    };

    assert_eq!(
        select_link_mode(&target, &features),
        Ok(MnnLinkMode::BuildFromSource)
    );
}

#[test]
fn windows_msvc_keeps_using_the_msvc_prebuilt() {
    let target = target("windows", "x86_64", "msvc", "x86_64-pc-windows-msvc");

    assert_eq!(
        select_link_mode(&target, &BuildFeatures::default()),
        Ok(MnnLinkMode::Prebuilt)
    );
    assert!(uses_msvc_flags(&target).unwrap());
}

#[test]
fn windows_gnu_never_uses_the_msvc_prebuilt_or_msvc_flags() {
    let target = target("windows", "x86_64", "gnu", "x86_64-pc-windows-gnu");

    assert_eq!(prebuilt_asset_name(&target, "dev"), None);
    assert_eq!(
        select_link_mode(&target, &BuildFeatures::default()),
        Ok(MnnLinkMode::BuildFromSource)
    );
    assert!(!uses_msvc_flags(&target).unwrap());
}

#[test]
fn windows_gnu_cuda_source_build_is_rejected_before_invoking_cmake() {
    let target = target("windows", "x86_64", "gnu", "x86_64-pc-windows-gnu");
    let features = BuildFeatures {
        cuda: true,
        ..BuildFeatures::default()
    };

    assert_eq!(
        select_link_mode(&target, &features),
        Err(BuildConfigError::UnsupportedBackendForTarget {
            backend: "cuda",
            target: "x86_64-pc-windows-gnu",
        })
    );
}

#[test]
fn user_provided_mnn_library_takes_precedence_over_backend_building() {
    let target = target("windows", "x86_64", "gnu", "x86_64-pc-windows-gnu");
    let features = BuildFeatures {
        cuda: true,
        mnn_dynamic: true,
        ..BuildFeatures::default()
    };

    assert_eq!(
        select_link_mode(&target, &features),
        Ok(MnnLinkMode::Dynamic)
    );
}

#[test]
fn mutually_exclusive_link_features_are_rejected() {
    let target = target("linux", "x86_64", "gnu", "x86_64-unknown-linux-gnu");
    let features = BuildFeatures {
        mnn_dynamic: true,
        mnn_static: true,
        ..BuildFeatures::default()
    };

    assert_eq!(
        select_link_mode(&target, &features),
        Err(BuildConfigError::ConflictingLinkModes)
    );
}

#[test]
fn unknown_windows_environment_is_rejected() {
    let target = target("windows", "x86_64", "", "x86_64-pc-windows-unknown");

    assert_eq!(
        select_link_mode(&target, &BuildFeatures::default()),
        Err(BuildConfigError::UnsupportedWindowsEnvironment(""))
    );
    assert_eq!(
        uses_msvc_flags(&target),
        Err(BuildConfigError::UnsupportedWindowsEnvironment(""))
    );
}

#[test]
fn ios_simulator_uses_the_simulator_asset() {
    let target = target("ios", "aarch64", "", "aarch64-apple-ios-sim");

    assert_eq!(
        prebuilt_asset_name(&target, "dev").as_deref(),
        Some("mnn-dev-ios-arm64-sim")
    );
}

#[test]
fn source_built_or_user_supplied_static_gpu_mnn_is_linked_whole_archive() {
    let vulkan = BuildFeatures {
        vulkan: true,
        ..BuildFeatures::default()
    };
    let cuda_static = BuildFeatures {
        cuda: true,
        mnn_static: true,
        ..BuildFeatures::default()
    };

    assert!(should_link_mnn_whole_archive(
        MnnLinkMode::BuildFromSource,
        &vulkan
    ));
    assert!(should_link_mnn_whole_archive(
        MnnLinkMode::Static,
        &cuda_static
    ));
}

#[test]
fn prebuilt_and_dynamic_mnn_do_not_force_whole_archive_linking() {
    let metal = BuildFeatures {
        metal: true,
        ..BuildFeatures::default()
    };

    assert!(!should_link_mnn_whole_archive(
        MnnLinkMode::Prebuilt,
        &metal
    ));
    assert!(!should_link_mnn_whole_archive(MnnLinkMode::Dynamic, &metal));
}
