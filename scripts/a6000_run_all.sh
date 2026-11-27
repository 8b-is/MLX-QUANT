#!/usr/bin/env bash
# One-command A6000 (49.13.101.247: RTX 6000 Ada 48 GB + Xeon Gold 5412U AVX-512)
# workflow implementing Phases 1 -> 2 -> 3 directly on the workstation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "=== [Step 1/3] Native AVX-512 + CUDA SM_89 Build, Verify & Benchmark ==="
export RUSTFLAGS="-C target-cpu=native"
export MLX_QUANT_CUDA_ARCH="${MLX_QUANT_CUDA_ARCH:-sm_89}"
export A6000_STATUS_URL="${A6000_STATUS_URL:-http://127.0.0.1:8790/a6000/api/status}"

mkdir -p out
cargo test --manifest-path mlx-quant-linux/Cargo.toml --workspace
cargo clippy --manifest-path mlx-quant-linux/Cargo.toml --workspace --all-targets -- -D warnings
cargo run --release --manifest-path mlx-quant-linux/Cargo.toml -- \
  --verify-all --bench --a6000-status | tee out/a6000_avx512_cuda_receipt.txt

echo "=== [Step 2/3] Scaled BeeHive-ERNIE #![no_std] Rust Coder Training (CUDA) ==="
PRESET="${A6000_PRESET:-125m}"
MAX_VRAM_GB="${A6000_MAX_VRAM_GB:-24}"
OUT_BUNDLE="${REPO_ROOT}/out/eightb-nostd-rust-coder-ternary-engram-v3"

PUSH_ARGS=()
if [[ -n "${A6000_PUSH_HF:-}" ]]; then
  PUSH_ARGS+=(--push-hf "${A6000_PUSH_HF}")
fi

python3 scripts/train_a6000_beehive_rust_coder_cuda.py \
  --preset "${PRESET}" \
  --max-vram-gb "${MAX_VRAM_GB}" \
  --out-dir "${OUT_BUNDLE}" \
  "${PUSH_ARGS[@]}"

echo "=== [Step 3/3] Post-Run Live a6000-orchestrator Telemetry ==="
cargo run --release --manifest-path mlx-quant-linux/Cargo.toml -- --a6000-status
