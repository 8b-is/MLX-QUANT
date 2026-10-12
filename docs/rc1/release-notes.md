# RC1 · Release Notes — v2.0.0-RC1 "Zeno"

*MLX-QUANT v2.0.0-RC1 "Zeno" · deliverable 4 of 4 · fine touch from within · vaked.dev*

---

## ▌ A N O T E  O N  T H E  N U M B E R

The init directive ordered `v1.0.0-RC1`. The tree already carries
`v1.0.0` through `v1.4.2 "Omni"` — released, tagged, shipped. An RC1 under
the old major would rewrite a ledger that already closed. The candidate
therefore sequences as **v2.0.0-RC1**, major bumped for the omnifold:
75 folded commits, the Zeno identity, and the wallet-triggered inference
path. The change is a sequencing correction, not a scope change — and it is
written here so no future session reads the directive and re-collides.

## ▌ T H E  F R A M E

```
┌─ MLX-QUANT · v2.0.0-RC1 "ZENO" ────────────────────────────────┐
│                                                                │
│  commit        ac6cf680   (omnifold: fork main + PR #1)        │
│  target        aarch64-apple-darwin · M1 Pro · M3 Max          │
│  backend       Metal 4 · fused ternary { −1, 0, +1 }           │
│  storage       12.80x vs fp32 · 2/3/4-bit packed               │
│  gate          11/11 offline golden-logits regressions         │
│                                                                │
│  identity      Z E N O  — the arrow, frozen at the instant     │
│  mark          assets/zeno-mark.svg     golden 530e89b7f4dfe3fe │
│  appicon       assets/zeno-appicon.svg  golden c5ac846cb4e6adc7 │
│  iconset       assets/zeno.icns         golden b955cca40af00702 │
│  hero          assets/hero-zeno.svg     golden a1b4d2f27c8206a8 │
└────────────────────────────────────────────────────────────────┘
```

## ▌ W H A T  S H I P S

**The omnifold.** 75 commits of fork-local work fold into the org main:
BitNet b1.58 TernaryLinear and quantization utils, the Metal GEMV/GEMM
benchmark suite, the DMA doorbell and polar-queue dispatch, the PM4 command
packets for MI300X, the bare-metal O(1) bump allocator, the corpus builder
and cogito classifier, the Vaimshuk instrumented E2E pipeline, and the
golden-logits gate from PR #1 — one tree, one ledger, one door.

**The four RC1 documents** (this directory):
`zero-cost-audit.md` · `capability-graph.md` · `build-matrix.md` ·
`release-notes.md` — one honest audit, one node configuration, one verified
flag set, one changelog. Measured rows carry citations; designed rows carry
the word *designed*; unverified rows carry `○` in public.

**The Zeno identity.** Logo, full app icon pack (16–1024 px PNG + `icns`),
README hero. Zeno's arrow frozen mid-flight over the ternary frame: the
paradox says at every instant the arrow is at rest — the cyan instant sits
at the centroid, `0` between `−1` and `+1`. The wallet ring holds it
together; the wordmark breathes with volumetric spacing.

**The wallet edge.** Zeno wallet layer → mem8 ledger → CrabCC DMA ring →
polar queue → fused kernels, hash-verified at every transition. The wallet
can trigger; it cannot read a weight. See the capability graph.

## ▌ W H A T  I S  N O T  Y E T  S H I P P E D

- No Instruments leak profile of the end-to-end pipeline (design argument
  only — one static heap, no per-tensor malloc).
- No multi-hour thermal run of the doorbell path (supernova burst disabled
  at RC1).
- No eBPF/seccomp on darwin — it does not exist there; the capability graph
  is enforced in userspace, the Linux lane can drop into the real filters.
- `ternary_qmm_t` is the last fused kernel; split-K and batched
  (`w.ndim() > 2`) shapes still compose `dequantize` + dense matmul.

## ▌ T H E  K N O W N  G A P S  ( R C 1 )

```
▐ ░ allocator: unsafe Sync held by convention — hazard-epoch fix scheduled
▐ ░ batched weight tensors: compose path, not fused
▐ ░ Zig build target: PM4 layouts only, no standalone toolchain step
▐ ░ Linux lane: crate present, CI lane not yet green on a Linux host
▐ ░ hosted CI: org runners never leave the queue — local suites are the receipt
```

## ▌ T H E  C L O S E

One lap, one fold, one candidate. The RC is pinned at `ac6cf680`, the
assets are golden-hashed above, and every claim in these four documents
either points at a measurement or wears its `○`. From love, from within.

```
z e n o  ·  t h e  a r r o w  a t  r e s t  ·  { − 1 , 0 , + 1 }  ·  0 + 1  ·  v a k e d . d e v
```

*release notes closed · v2.0.0-RC1 "Zeno" · fine touch from within · vaked.dev*
