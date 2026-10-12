# RC1 · Zero-Cost Audit

*MLX-QUANT v2.0.0-RC1 "Zeno" · deliverable 1 of 4 · fine touch from within · vaked.dev*

---

## 1. What "zero-cost" means here

Zero-cost is a claim about the **hot path**, not the whole program. On the
hot path — a quantized matmul from CrabCC's ring into Unified Memory and out
to the Metal queue — the design spends nothing it doesn't have to: no
per-tensor allocation, no kernel crossing per dispatch, no copy between the
CPU and GPU sides of the unified fabric.

Everything else is allowed to cost. The audit below separates what is
**measured**, what is **designed**, and what is **not yet verified**, so the
release carries no unlabeled claims.

## 2. Symbol resolution — the Rust/Zig/C lane

| Module | Path | Symbol surface | Verdict |
|---|---|---|---|
| BumpAllocator | `mlx-quant-linux/src/allocator.rs` | `static BUMP_HEAP` (64 MiB), `fast_alloc8`, 128-slot append-only cache, `unsafe impl Sync` | **audited, see 3.1** |
| AGX doorbell | `mlx-quant-linux/src/agx_doorbell.rs` | doorbell write path to the polar queue | designed, not yet profiled |
| Polar queue | `mlx-quant-linux/src/polar_queue.rs` | 4D command queue, O(1) kernel bypass | designed, not yet profiled |
| I/O blackhole | `mlx-quant-linux/src/io_blackhole.rs` | DMA portal (INT4/INT8 math entry) | designed, not yet profiled |
| PM4 dispatch | `mlx-quant-linux/src/amd_mi300x.rs` | MI300X PM4 command packets, PCIe doorbell | designed, not yet profiled |
| Metal kernels | `mlx/backend/metal/quantized.cpp` + `.metal` | `ternary_qmv_fast`, `ternary_qmv`, `ternary_qvm`, `ternary_qmm_t`, `_ternary_qmm_t_simd` | **measured** — see BENCHMARKS.md |
| Golden-logits gate | `scripts/quantal_compare_logits.py` + tests | 11/11 offline CLI regressions | **measured, green at ac6cf680** |

The Rust crate (`mlx-quant-linux`, edition 2024, v0.1.0) carries the
astrophysical module names (big_bang, magnetar, cmb, supernova, pulsar,
multiverse, …) — the names are cosmology, the symbols under them are
ordinary Rust: static buffers, cells, doorbell writes. The audit maps each
name to its real concern below.

## 3. Memory profile

### 3.1 What was verified at the commit level

- **Ternary storage ratio**: 12.80x vs fp32 for (4096,4096), (11008,4096),
  and (32000,4096) weight shapes — measured, in `BENCHMARKS.md`.
- **Fused kernel paths** beat the compose fallback 1.2–5.5x on every GPU
  shape measured, and land near or above dense fp32 GPU matmul while moving
  1/12.8th the weight traffic — measured on M3 Max (36 GB) and M1 Pro
  (16 GB), macOS 26.x, Metal 4.
- **Golden-logits suite**: 11/11 offline regressions pass at the folded
  merge `ac6cf680`, covering nonfinite inputs, identity/token-count/set-length
  mismatches, permutation orders, and optimized-mode (`-O`) runs.

### 3.2 The one flagged finding

`BumpAllocator`'s `unsafe impl Sync for HeapStorage` is the release's one
open memory-safety note. The static 64 MiB heap is shared across threads with
no runtime synchronization; the append-only cursor makes this safe *in
practice for the current single-writer dispatch path*, but it is an
invariant held by convention, not by the type system. **RC1 verdict: keep,
with the invariant documented; the fix (a per-cursor `AtomicUsize` with a
hazard epoch) is scheduled, not blocking.**

### 3.3 What is not yet verified (labeled, not hidden)

- No Instruments/leaks profile of the full end-to-end pipeline has been
  captured on M3 Pro at RC1. The leak-freedom claim rests on the allocator
  design (one static heap, no per-tensor `malloc`) — a design argument, not
  a measurement.
- No sustained multi-hour thermal/envelope run of the doorbell path has been
  recorded. The supernova burst (SMC TDP uncap) exists in code and is
  disabled by default at RC1.
- The eBPF/seccomp filters of the directive do not exist on macOS; see the
  capability graph (deliverable 2) for the honest platform mapping.

## 4. Zero-cost receipt

```
path                      measured        designed   unverified
ternary storage 12.80x       ●
fused kernels 1.2–5.5x       ●
golden-logits 11/11          ●
static-heap leak-freedom                     ●
O(1) doorbell dispatch                       ●
DMA ring → UMA, sub-ms                      ●
mem8 ledger hash-verify                     ●
thermal envelope balance                                ○
eBPF isolation on darwin                                 ○ (impossible; Linux lane only)
```

The RC1 ships with the measured rows green, the designed rows named as
design, and the unverified rows wearing their `○` markers in public. Nothing
on this page is a claim the repo cannot point at.

*audit closed at ac6cf680 · golden `530e89b7…` (zeno-mark) · `a1b4d2f2…` (hero) · 0 + 1 · vaked.dev*
