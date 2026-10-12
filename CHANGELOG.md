# Changelog

MLX-QUANT is a standalone fork of [ml-explore/mlx](https://github.com/ml-explore/mlx)
maintained by [8b-is](https://github.com/8b-is). It is not intended to be
upstreamed. Versions here are independent of upstream MLX's own version
numbers (this fork branched from upstream `973e27f8`, tagged `v0.32.0` in
upstream's own scheme).

## v2.0.0-RC1 "Zeno"

Release candidate for the 2.0 line. Major bump: the omnifold (75 commits of
fork-local work folded onto org main alongside the golden-logits gate),
the Zeno identity (logo, full app icon pack, README hero), and the
wallet-triggered inference path mapped onto the Vaked capability graph.

- **The omnifold** — `ac6cf680`: fork main + PR #1 (golden-logits
  validation) in one tree. Includes BitNet b1.58 `TernaryLinear` +
  quantization utils, fused Metal kernels (`ternary_qmv_fast`/`ternary_qmv`/
  `ternary_qvm`/`ternary_qmm_t`), the GEMV/GEMM + memory-bandwidth bench
  suite, DMA doorbell / polar-queue dispatch, PM4 command packets for
  MI300X, the bare-metal O(1) bump allocator, the corpus builder + cogito
  classifier, and the Vaimshuk instrumented E2E pipeline.
- **Zeno identity** — `assets/zeno-mark.svg`, `assets/zeno-appicon.svg`,
  `assets/zeno.icns`, `assets/icons/zeno-{16..1024}.png`,
  `assets/hero-zeno.{svg,png}`. The arrow frozen at the instant: the ternary
  frame {−1, 0, +1} pierced mid-flight, the cyan zero at the centroid,
  the wallet ring around it.
- **The four RC1 documents** — `docs/rc1/zero-cost-audit.md`,
  `docs/rc1/capability-graph.md`, `docs/rc1/build-matrix.md`,
  `docs/rc1/release-notes.md`. Measured rows cite sources; designed rows
  say *designed*; unverified rows carry `○` in public.
- **Retired** the pre-RC1 hero/infographic assets (hero.jpg,
  mlx-quant-hero.png, mlx-quant-ternary-diagram.png, the landing
  infographics) ahead of the Zeno identity.

Known gaps at RC1 (see `docs/rc1/release-notes.md` for the full frame):
BumpAllocator's `unsafe impl Sync` is held by convention (hazard-epoch fix
scheduled); batched (`w.ndim() > 2`) weights still compose instead of fuse;
no standalone Zig build target; the Linux lane has no green CI host yet;
org-hosted CI never leaves the queue — local suites are the receipt.

## v1.4.2 "Omni"

Closes essentially every remaining gap named in v1.0.0's "known gaps"
section. Ternary quantized ops on GPU now have real fused kernel coverage
for every shape except a batched (`w.ndim() > 2`) weight tensor, which
still composes `dequantize` + a dense matmul.

### GPU (Metal) backend — new since v1.0.0

- **`gather_qmm` (MoE-style indexed matmul)** now works for
  `mode="ternary"` on both CPU and GPU — it previously threw
  unconditionally. Composed from `dequantize` + the existing `gather_mm`
  op (which already implements per-token/per-expert weight selection
  generically), verified for correctness and gradients against
  `mx.gather_mm` on dequantized weights.
- **`ternary_qvm`**: fused kernel for `quantized_matmul(transpose=False)`,
  non-batched weights. Mirrors `qvm_impl`.
- **`ternary_qmv`**: general, bounds-checked fused kernel for
  `transpose=True` covering any `K`/`N` (safe zero-padded/partial-byte
  K-tail, "slide the last tile back" trick for `N` not a multiple of 8) —
  the fallback whenever `ternary_qmv_fast`'s exact-multiple precondition
  doesn't hold. Mirrors `fp_qmv_impl`. Between this and `ternary_qmv_fast`,
  `transpose=True` non-batched matmul is now *always* fused.
- **`ternary_qmm_t`**: tiled GEMM for large-M matmul (`M >= 32`,
  non-batched weights, `transpose=True`), decoding each weight group once
  per 32-row tile and reusing it across all 32 rows via threadgroup
  memory instead of re-decoding per row. A real `steel::BlockMMA`/
  `BlockLoader` integration mirroring `QuantizedBlockLoader`/`qmm_t_impl`/
  `affine_qmm_t`. Measured 1.25–1.7x faster than the compose fallback
  across `M=32..256`, landing close to dense fp32 GPU matmul's own
  timing. `qmm_splitk` (the very-large-K split-K variant) is not
  implemented — `ternary_qmm_t` already handles any `K` correctly, just
  without split-K's extra parallelism.
- Each of the four fused paths above (`qmv_fast`, `qvm`, `qmv`, `qmm_t`)
  is dispatched through its own dedicated `fast::Custom`-derived
  primitive (`TernaryQmvFast`/`TernaryQvm`/`TernaryQmv`/`TernaryQmm`)
  rather than through `QuantizedMatmul`'s shared affine/fp dispatch tree,
  so adding each one could never leave some shape/dtype combination
  reaching a kernel name that was never written, and none of them touch
  or risk the already-working affine/fp code paths.

### Fixed

- **A real CPU-only (no Metal, no CUDA) build regression**, present since
  `TernaryQmvFast` was introduced in v1.0.0: `mlx/backend/no_gpu/
  primitives.cpp` provides `NO_GPU_MULTI` stub `eval_gpu` definitions for
  every GPU-only `fast::Custom` primitive so their vtables link when no
  GPU backend is compiled at all; the ternary primitives were never added
  there, so the CPU-only build silently failed to link `tests/tests`.
  (Earlier "CPU-only build clean" checks in this project's own history
  were false positives from stale incremental linking — ninja hadn't
  actually relinked the test binary. Verification now always forces a
  fresh link first.)
- `get_quantized_kernel_wrapped`/`get_qmm_nax_kernel_wrapped`'s kernel-name
  computation only ever branched `affine_`/`fp_`, silently mapping
  ternary to the wrong `fp_` prefix. Harmless for the default non-JIT
  build; would have broken `MLX_METAL_JIT=ON` builds.

### Numerical property documented (not a bug)

Unlike `affine`'s min/max-based scale (order-independent — min and max
never disagree regardless of reduction order), ternary's scale is
`mean(|w|)`, a sum, and floating-point addition is not associative. The
GPU kernels' `simd_sum` (a parallel tree reduction) can legitimately
disagree with a sequentially-computed reference sum in the last bit,
which occasionally (~2% of random seeds in testing) flips a weight
whose `w/scale` ratio sits almost exactly on a `round()` tie. Bounded to
one code step, and — new observation from `ternary_qmm_t` testing — its
visible blast radius scales with `M` for GEMM-style kernels, since a
single mis-quantized weight is shared by every output row that reads
that weight column (GEMV-style kernels only ever showed it in one row).
Documented in `ternary_quantized.h` and handled by
`assert_ternary_gpu_allclose` (tolerates a small, bounded fraction of
outliers) in the test suite, not by chasing the GPU reduction order to
match one arbitrary reference implementation.

### Known gaps / explicitly out of scope for v1.4.2

- No Swift bindings (`mlx-swift`) — still C++/Python only. Consuming
  this from a Swift project (e.g. Osaurus) needs a separate port.
- Batched weights (`w.ndim() > 2`) for `quantized_matmul` still compose
  rather than using a fused kernel, for both `transpose` settings.
- `qmm_splitk`/`qvm_split_k` (very-large-K split-K variants) not
  implemented — `ternary_qmm_t`/`ternary_qvm` already handle any `K`
  correctly without them.
- `qmv_wide`/`qmv_quad` (further GEMV speed refinements for specific
  `M`/`K` ranges) not implemented — `ternary_qmv`'s general path already
  covers those shapes correctly, just without their extra optimization.

## v1.0.0

First tagged release of this fork's headline feature: a native **ternary
("BitNet b1.58") quantization mode** — 2-bit codes `{0, 1, 2}` decoding to
signed `{-1, 0, 1}` via `code - 1`, one real-valued scale per group (no
bias), 16 codes packed per `uint32` word LSB-first. Usable via the existing
`mx.quantize`/`mx.dequantize`/`mx.quantized_matmul` API with
`mode="ternary"`, `bits=2`.

### CPU backend

- Native ternary quantize/dequantize and `quantized_matmul`, wired through
  the same primitive/dispatch machinery as `"affine"` and the `"mxfp4"`/
  `"mxfp8"`/`"nvfp4"` family.
- Fused, M-tiled SIMD `quantized_matmul` kernel (`mlx/backend/cpu/
  quantized.cpp`): decodes packed weight words once per output column per
  M-tile (`MTILE=4`) and reuses them across all 4 rows' accumulators,
  instead of unpacking per row. Falls back to a correctness-first
  unpack+BLAS path for `transpose=false`, group sizes that don't divide a
  whole packed word, and `bfloat16` (no real SIMD width on this backend).
- Measured (see `benchmarks/python/ternary_qmm_bench.py`): 2.3–2.4x faster
  than this codebase's own best comparable CPU kernel (`affine`, bits=4)
  and 8–17x faster than `affine` bits=2 (no SIMD path in this codebase at
  all). Still ~14–18x slower than dense fp32 BLAS — an honest gap shared
  by every hand-rolled, non-multi-threaded CPU kernel in this codebase,
  not specific to ternary.

### GPU (Metal) backend

- Real Metal kernels for standalone `quantize`/`dequantize`
  (`mlx/backend/metal/kernels/ternary_quantized.h`), covering group sizes
  32/64/128 and `float`/`float16`/`bfloat16`, adapted from `affine_quantize`'s
  generic per-group SIMD reduction (mean-of-abs instead of min/max range,
  no bias).
- A fused `ternary_qmv_fast` kernel for the common `nn.Linear`/decode shape
  (non-batched weights, `transpose=True`, `N % 8 == 0`, `K % 1024 == 0`),
  mirroring `fp_qmv_fast_impl`'s threading model. Dispatched through a
  dedicated `fast::TernaryQmvFast` primitive rather than through
  `QuantizedMatmul`'s shared dispatch tree, so it can't leave any
  shape/dtype combination reaching a kernel name that doesn't exist.
  Measured: 2.9–6.1x faster than the compose fallback below, and 1.6–2.8x
  faster than dense fp32 GPU matmul at `M=1`.
- Every other `quantized_matmul` shape/setting (batched weights,
  `transpose=False`, or either divisibility check failing) composes
  `dequantize` + the existing dense GPU matmul instead of a fused kernel —
  correct for any shape, at the cost of not being fused. Gradients flow
  through this same fallback (`fast::Custom`'s fallback-based vjp/jvp), so
  autodiff works for the fused path too without a bespoke implementation.
- **Not implemented on GPU**: a general (non-`qmv_fast`) tiled matmul
  kernel family (`qmv`/`qmv_wide`/`qmm`/`qmm_splitk`/`qvm`/`qvm_split_k`) —
  those shapes use the compose fallback, not a fused kernel.
  `gather_qmm` (MoE-style indexed matmul) still raises for `mode="ternary"`
  on both CPU and GPU — out of scope for this release.

### Known gaps / explicitly out of scope for v1.0.0

- No Swift bindings (`mlx-swift`) — this release covers the C++/Python
  fork only. Consuming this from a Swift project (e.g. Osaurus) would
  require a separate port, not started.
- No `gather_qmm` (MoE) support for `mode="ternary"`.
- No fused GPU kernel beyond the single `qmv_fast` shape described above.

### Testing

`python/tests/test_quantized.py`: quantize/dequantize round-trip and
`quantized_matmul` correctness against an independent RoundClip reference,
covering CPU and GPU, all 3 group sizes, `float32`/`float16`/`bfloat16`,
both `transpose` settings, batched activations, and every GPU fast-path
fallback boundary (`K` not a multiple of 1024, `N` not a multiple of 8,
batched weights). Full `mlx` C++ test suite and full `python/tests` suite
pass on both a Metal-enabled and a CPU-only build.
