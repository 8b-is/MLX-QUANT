use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn run_checked(cmd: &mut Command, label: &str) {
    let status = cmd
        .status()
        .unwrap_or_else(|err| panic!("failed to spawn {label}: {err}"));
    assert!(status.success(), "{label} failed with status {status}");
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_metal_bridge)");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=shaders/mlx_quant_beehive.metal");
    println!("cargo:rerun-if-changed=shaders/ternary_beehive.wgsl");
    println!("cargo:rerun-if-changed=shaders/ternary_beehive.cu");
    println!("cargo:rerun-if-changed=ffi/metal_bridge.m");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "macos" {
        return;
    }

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let clang_arch = match target_arch.as_str() {
        "aarch64" => "arm64",
        "x86_64" => "x86_64",
        other => panic!("unsupported macOS target arch for metal_bridge: {other}"),
    };

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    let metal_src = manifest_dir.join("shaders/mlx_quant_beehive.metal");
    let air_out = out_dir.join("mlx_quant_beehive.air");
    let metallib_out = out_dir.join("mlx_quant_beehive.metallib");

    run_checked(
        Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", "-O2", "-c"])
            .arg(&metal_src)
            .arg("-o")
            .arg(&air_out),
        "xcrun metal compile",
    );

    run_checked(
        Command::new("xcrun")
            .args(["-sdk", "macosx", "metallib"])
            .arg(&air_out)
            .arg("-o")
            .arg(&metallib_out),
        "xcrun metallib link",
    );

    let bridge_src = manifest_dir.join("ffi/metal_bridge.m");
    let bridge_obj = out_dir.join("metal_bridge.o");
    let bridge_lib = out_dir.join("libmetal_bridge.a");

    run_checked(
        Command::new("xcrun")
            .args([
                "clang",
                "-fobjc-arc",
                "-O2",
                "-Wall",
                "-Wextra",
                "-arch",
                clang_arch,
                "-c",
            ])
            .arg(&bridge_src)
            .arg("-o")
            .arg(&bridge_obj),
        "xcrun clang metal_bridge.m",
    );

    if bridge_lib.exists() {
        let _ = fs::remove_file(&bridge_lib);
    }

    run_checked(
        Command::new("ar")
            .args(["rcs"])
            .arg(&bridge_lib)
            .arg(&bridge_obj),
        "ar libmetal_bridge.a",
    );

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=metal_bridge");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-cfg=has_metal_bridge");
}
