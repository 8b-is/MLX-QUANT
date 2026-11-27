# Zero-Cost Audit — RC1 Baseline & v3.0.0-alpha.1 Resolution

*MLX-QUANT v2.0.0-RC1 "Zeno" baseline + v3.0.0-alpha.1 "Ultra-OmniLLL Bare-Metal Substrate" · vaked.dev*

---

## 1. What "zero-cost" means here

Zero-cost is a claim about the **hot path**, not the whole program. On the
hot path — a quantized matmul from the epoch-scoped arena into Unified Memory
and out to NEON/Metal — the design spends nothing it doesn't have to: no
per-tensor heap `malloc`/`Vec`, no floating-point drift in core integer/Q15.16
oracles, and lock-free `AtomicUsize` CAS bump allocation.

Everything else is allowed to cost. The audit below strictly separates what is
**Measured**, what is **Compiled / Cross-Checked**, and what is **Designed**
so the release carries no unlabeled claims.

---

## 2. v3.0.0-alpha.1 "Ultra-OmniLLL Bare-Metal Substrate" Update

### 2.1 RC1 Findings Resolved

1. **RC1 §3.2 `BumpAllocator` `unsafe impl Sync` / `Cell` hazard — RESOLVED**
   - Removed `static BUMP_HEAP` (`UnsafeCell<[u8; 64 MiB]>`), `Cell`, and raw `*mut u8` tensor pointers.
   - `mlx-quant-core::arena::{EpochBumpArena, EpochBumpArenaSlice}` (`#![no_std]`) and `mlx-quant-linux::allocator::BumpAllocator` now own a heap-allocated `64`-byte-aligned backing buffer (`64 MiB` default via `BumpAllocator::new()`, configurable via `BumpAllocator::with_capacity`) freed in `Drop`.
   - Shared `alloc_bytes(&self, ...)` executes lock-free `AtomicUsize` CAS (`try_update(AcqRel, Acquire)`) yielding strictly disjoint `&mut [u8]` sub-slices tied to `&self`, while `reset_epoch(&mut self)` requires exclusive `&mut self` — statically preventing use-after-free across threads and live `Tensor<'a>` borrows.
   - Verified under an **8-thread zero-overlap concurrency stress test** in both `mlx-quant-core` and `mlx-quant-linux` (`--verify-all` and `cargo test`), with 0 overlapping ranges and clean epoch reclamation.
2. **`src/math/matmul.rs` `"SIMULATED INLINE ASM BLOCK"` stub — RESOLVED**
   - Replaced the RC1 stub with real cross-arch dispatch via `mlx-quant-asm`:
     - `ternary_2bit_gemv_neon` (AArch64 NEON + `FEAT_DotProd` `sdot` / `smull` fallback + `prfm pldl1keep`)
     - `int4_gemm_neon` (`2x4` tiled signed-nibble `sdot`/baseline NEON with `shl #4`/`sshr #4` sign extension + odd-`k` tail)
     - `hadamard_wht_16_neon` (`sqadd`/`sqsub` 4-stage saturating Walsh-Hadamard butterfly)
     - `bitcos_gemv` (L1-prefetched BITCOS sparse presence+sign GEMV)
   - Verified bit-for-bit against `mlx-quant-core` scalar oracles across 10,000-iteration SplitMix64 fuzzing, `i8::MIN`/`i8::MAX`, and odd dimensions (`19x73`, `17x13x73`).
3. **Metal `.metallib` C/Obj-C runtime bridge (`mlx-quant-gpu`) — MEASURED LIVE ON APPLE M1 MAX**
   - `ffi/metal_bridge.m` (Objective-C + ARC, zero warnings under `-Wall -Wextra`) compiles `shaders/mlx_quant_beehive.metal` into `$OUT_DIR/mlx_quant_beehive.metallib` via `xcrun -sdk macosx metal` + `metallib`.
   - `AgxDoorbell` and `PolarQueue` wrap `Option<mlx_quant_gpu::MetalRuntime>` (`"Apple Metal Live GPU"` when present, `"Portable Ring Simulator"` fallback otherwise) with atomic `ring_counter` and `last_gpu_ns`.
   - Verified **bit-for-bit parity** on `Apple M1 Max` (`macOS 27.2`, `cntfrq_el0 = 24,000,000 Hz`) against `mlx-quant-core` and `mlx-quant-asm` for `256x512` and `19x73` `ternary_2bit_gemv_i8`, `bitcos_gemv_i8`, and `tequila_deadzone_ste_q16`.
4. **All 29 Rust seed modules + `src/lib.rs` (`mlx-quant-linux`) — RESOLVED**
   - Replaced all RC1 stubs/floats (`Vec<usize>`, `f32`, `String`, `write_volatile`, zeroing mocks) with zero-float integer/`Ratio`/`Q15.16` implementations:
     - `IoBlackhole` bounded reader with EOF zero-padding
     - `Redshift` real AbsMean `Ternary2Bit`, AbsMax `Int4Packed`, and `BitcosSparse` quantizers
     - `FusionPolarity` real fused `H-BitLinear` forward (`hadamard_wht_normalized_q16` -> `quantize_activations_int4_symmetric` -> `ternary_2bit_gemv`)
     - `AmdComputeRing` wrapping 64-byte aligned `Pm4Ring<1024>` (`AcquireMem`, `DispatchDirect`, `fnv1a_active_digest`, `drain`)
     - `Antigravity` (`step_by(64)` `prefetch_l1`), `Pulsar` (`cntvct_el0`/`cntfrq_el0`), `DarkEnergy` (`Ratio` entropy), `CosmicMicrowaveBackground<CAP>` (circular Q15.16 KV ring), `EventHorizon` (overflow-safe bounds firewall), `GravitationalLensing` (fixed-array speculative prefix check), `HawkingRadiation` (`&mut BumpAllocator` epoch reset), and `TheGuardians` (arena/firewall integrity check).
5. **Upstream `ml-explore/mlx` `v0.32.5` Audit Check**
   - Checked `origin/main` at `fa1e016` (`ci: enforce blacklist against Blacksmith runners and actions`) against fetched upstream tag `v0.32.5` (`cfc675e`, `Fix macOS 14 build (#4960)`). Upstream audited cleanly without clobbering fork-local ternary / `v2.1.0` / `v2.2.0` / `v3.0.0-alpha.1` code.

### 2.2 v3.0.0-alpha.1 Measured, Compiled/Cross-Checked, and Designed Matrix

Host receipt: `Apple M1 Max`, `macOS 27.2` (`BuildVersion 26B5112a`), `ISA = aarch64-neon-dotprod`, `cntfrq_el0 = 24,000,000 Hz`.

#### Measured (executed live and verified on host)

- **Workspace tests (`cargo test --workspace`)**: `26 passed, 0 failed` (`mlx-quant-core`: 6, `mlx-quant-asm`: 8, `mlx-quant-gpu`: 6, `mlx-quant-linux`: 6), plus `0` warnings under `cargo clippy --workspace --all-targets -- -D warnings`.
- **Release self-check (`cargo run --release -- --verify-all --bench`)**: `13/13` parity & subsystem checks PASS:
  - `ternary_2bit_gemv` (`256x512`, `19x73`): Metal GPU == NEON ASM == Core Scalar
  - `bitcos_gemv` (`256x512`, `19x73`): Metal GPU == ASM == Core Scalar
  - `int4_gemm` (`32x64x64`, `17x13x73`): NEON ASM == Core Scalar
  - `tequila_ste_q16` (`256x512`, `19x73`): Metal GPU == Core Scalar
  - `h_bitlinear_wht16_and_forward`, `engram_table_q16`, `mhc_sinkhorn_4x4` (`max_birkhoff_err_q16 = 1`), `ramanujan_expander` (`192` edges), `mushroom_body_wta` (`sum_q16 = 65536`), `pm4_ring_1024`, `bump_allocator_8_thread_zero_overlap` (`128` disjoint allocs).
- **Ternary 2-bit GEMV Benchmarks (`--release`, `cntvct_el0` + Metal `GPUStartTime/GPUEndTime`)**:
  - `1024x1024` (`2,097,152` ops, `267,264` bytes):
    - `Scalar`: `170.25 us` (`12.32 GOP/s`, `1.57 GB/s`, Parity `OK`)
    - `aarch64-neon-dotprod`: `51.08 us` (`41.06 GOP/s`, `5.23 GB/s`, Parity `OK`)
    - `Apple Metal Live GPU`: wall `993.42 us` (`2.11 GOP/s`, `0.27 GB/s`, includes shared buffer copy+sync); GPU kernel-only `17.21 us` (`121.84 GOP/s`, `15.53 GB/s`, Parity `OK`)
  - `4096x1024` (`8,388,608` ops, `1,065,984` bytes):
    - `Scalar`: `693.04 us` (`12.10 GOP/s`, `1.54 GB/s`, Parity `OK`)
    - `aarch64-neon-dotprod`: `214.49 us` (`39.11 GOP/s`, `4.97 GB/s`, Parity `OK`)
    - `Apple Metal Live GPU`: wall `984.60 us` (`8.52 GOP/s`, `1.08 GB/s`); GPU kernel-only `31.03 us` (`270.30 GOP/s`, `34.35 GB/s`, Parity `OK`)
- **In-memory PM4 ring encoding (`Pm4Ring<1024>`)**: 64-byte alignment, wraparound, invalid-count/header rejection, and FNV-1a active digest verified live.

#### Compiled / Cross-Checked (checked cleanly on non-host targets, not runtime-executed on those CPUs)

- `mlx-quant-core`: `cargo check --target aarch64-unknown-none` and `cargo check --target aarch64-unknown-linux-musl` (`#![no_std]`, 0 warnings with `RUSTFLAGS=""`).
- `mlx-quant-asm`: `cargo check --target aarch64-unknown-none` and `RUSTFLAGS="" cargo check --target x86_64-apple-darwin` (0 warnings; covers AVX2 & AVX-512BW+VNNI codegen paths).
- `mlx-quant-gpu --no-default-features`: `cargo check --target aarch64-unknown-none` (clean `#![no_std]` `Pm4Ring` build skipping macOS Metal bridge cleanly).

#### Designed (architectural descriptors / opt-in / non-host hardware paths)

- **Apple Silicon AMX**: `AmxTilePlan` (`32x32x32`) and `.inst` opcode encodings compiled; hardware `.inst` probe is gated behind `feature = "apple-amx"` and disabled by default.
- **x86_64 AVX2 / AVX-512BW+VNNI runtime**: cross-compiled cleanly; runtime execution not measured on Apple Silicon host.
- **WebGPU WGSL (`ternary_beehive.wgsl`) & CUDA (`ternary_beehive.cu`)**: embedded shader sources present via `include_str!`; runtime GPU dispatch not executed on host.
- **AMD MI300X PCIe BAR2 MMIO doorbell**: `Pm4Ring<1024>` packet construction is measured in memory; physical PCIe BAR2 MMIO submission is `DesignedDescriptor`.
- **Linux SMC TDP uncap / RDMA RoCEv2 / `mlock` / P-core affinity / eBPF**: honest userspace `DesignedDescriptor` structs only (`Supernova` burst disabled by default; `Magnetar` `locked: false` / `pinned: false`).

---

## 3. Historical RC1 Baseline Record (v2.0.0-RC1 "Zeno")

| Module | Path | RC1 Symbol surface | RC1 Verdict -> v3.0.0-alpha.1 Status |
|---|---|---|---|
| BumpAllocator | `mlx-quant-linux/src/allocator.rs` | `static BUMP_HEAP` (64 MiB), `Cell`, `unsafe impl Sync` | **RC1 flagged -> RESOLVED in v3.0.0-alpha.1** (`EpochBumpArenaSlice` CAS + `&mut self` epoch reset) |
| MatMul / ASM | `mlx-quant-linux/src/math/matmul.rs` | `"SIMULATED INLINE ASM BLOCK"` | **RC1 stub -> RESOLVED in v3.0.0-alpha.1** (`mlx-quant-asm` NEON/DotProd kernels) |
| AGX doorbell | `mlx-quant-linux/src/agx_doorbell.rs` | mocked MMIO pointer | **RESOLVED in v3.0.0-alpha.1** (live `MetalRuntime` `.metallib` + simulator fallback) |
| Polar queue | `mlx-quant-linux/src/polar_queue.rs` | `Vec`/`String` 4D queue stub | **RESOLVED in v3.0.0-alpha.1** (bounded 8-arm arena queue + GPU dispatch) |
| I/O blackhole | `mlx-quant-linux/src/io_blackhole.rs` | no-op dummy tensor alloc | **RESOLVED in v3.0.0-alpha.1** (bounded `Read`/`File` loader + EOF zero-pad) |
| PM4 dispatch | `mlx-quant-linux/src/amd_mi300x.rs` | raw mock `AmdPacket` | **RESOLVED in v3.0.0-alpha.1** (`Pm4Ring<1024>` in-memory measured; MMIO designed) |
| Metal kernels (C++) | `mlx/backend/metal/quantized.cpp` + `.metal` | `ternary_qmv_fast`, `ternary_qmv`, `ternary_qvm`, `ternary_qmm_t` | **measured** — see `BENCHMARKS.md` |
| Golden-logits gate | `scripts/quantal_compare_logits.py` + tests | 11/11 offline CLI regressions | **measured, green at `ac6cf680`** |

```
path                                      measured   cross-checked   designed
ternary storage 12.80x (RC1)                 ●
fused Metal kernels 1.2–5.5x (RC1)           ●
golden-logits 11/11 (RC1)                    ●
BumpAllocator CAS + 8-thread zero-overlap    ● (RC1 hazard resolved)
NEON/DotProd 2-bit GEMV & INT4 GEMM & WHT16  ●
Live Metal .metallib (2-bit/BITCOS/Tequila)  ● (Apple M1 Max)
FusionPolarity H-BitLinear & BeeHive suite   ●
Pm4Ring<1024> in-memory packet encoding      ●
no_std / musl / x86_64 compilation                        ●
AMX / AVX runtime / WGSL / CUDA / PCIe MMIO / RDMA / SMC               ●
```

*v3.0.0-alpha.1 audit updated against `origin/main` `fa1e016` and upstream `v0.32.5` (`cfc675e`) · vaked.dev*
