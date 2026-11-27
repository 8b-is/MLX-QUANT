use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn run_checked(cmd: &mut Command, label: &str) {
    let status = cmd
        .status()
        .unwrap_or_else(|err| panic!("failed to spawn {label}: {err}"));
    assert!(status.success(), "{label} failed with status {status}");
}

fn find_nvcc() -> Option<(PathBuf, Option<PathBuf>)> {
    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        let candidate = PathBuf::from(&cuda_home).join("bin/nvcc");
        if candidate.is_file() {
            let lib64 = PathBuf::from(&cuda_home).join("lib64");
            return Some((candidate, lib64.is_dir().then_some(lib64)));
        }
    }

    if let Ok(path_var) = env::var("PATH") {
        for dir in env::split_paths(&path_var) {
            let candidate = dir.join("nvcc");
            if candidate.is_file() {
                let lib64 = candidate
                    .parent()
                    .and_then(Path::parent)
                    .map(|root| root.join("lib64"))
                    .filter(|p| p.is_dir());
                return Some((candidate, lib64));
            }
        }
    }

    let fallback = PathBuf::from("/usr/local/cuda/bin/nvcc");
    if fallback.is_file() {
        let lib64 = PathBuf::from("/usr/local/cuda/lib64");
        return Some((fallback, lib64.is_dir().then_some(lib64)));
    }
    None
}

fn build_macos_metal(manifest_dir: &Path, out_dir: &Path, target_arch: &str) {
    let clang_arch = match target_arch {
        "aarch64" => "arm64",
        "x86_64" => "x86_64",
        other => panic!("unsupported macOS target arch for metal_bridge: {other}"),
    };

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

fn build_linux_cuda(manifest_dir: &Path, out_dir: &Path) {
    let Some((nvcc, lib64)) = find_nvcc() else {
        return;
    };
    let ok = Command::new(&nvcc)
        .arg("--version")
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        return;
    }

    let arch = env::var("MLX_QUANT_CUDA_ARCH")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "sm_89".to_string());

    let bridge_cu = manifest_dir.join("ffi/cuda_bridge.cu");
    let bridge_obj = out_dir.join("cuda_bridge.o");
    let bridge_lib = out_dir.join("libcuda_bridge.a");

    run_checked(
        Command::new(&nvcc)
            .args(["-O3", &format!("-arch={arch}"), "-Xcompiler", "-fPIC", "-c"])
            .arg(&bridge_cu)
            .arg("-o")
            .arg(&bridge_obj),
        "nvcc cuda_bridge.cu",
    );

    if bridge_lib.exists() {
        let _ = fs::remove_file(&bridge_lib);
    }

    run_checked(
        Command::new("ar")
            .args(["rcs"])
            .arg(&bridge_lib)
            .arg(&bridge_obj),
        "ar libcuda_bridge.a",
    );

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=cuda_bridge");
    if let Some(dir) = lib64 {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-cfg=has_cuda_bridge");
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_metal_bridge)");
    println!("cargo:rustc-check-cfg=cfg(has_cuda_bridge)");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=shaders/mlx_quant_beehive.metal");
    println!("cargo:rerun-if-changed=shaders/ternary_beehive.wgsl");
    println!("cargo:rerun-if-changed=shaders/ternary_beehive.cu");
    println!("cargo:rerun-if-changed=ffi/metal_bridge.m");
    println!("cargo:rerun-if-changed=ffi/cuda_bridge.cu");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=MLX_QUANT_CUDA_ARCH");

    if env::var_os("CARGO_FEATURE_STD").is_none() {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    if target_os == "macos" {
        build_macos_metal(&manifest_dir, &out_dir, &target_arch);
    } else if target_os == "linux" {
        build_linux_cuda(&manifest_dir, &out_dir);
    }
}
