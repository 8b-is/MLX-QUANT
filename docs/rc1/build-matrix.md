# RC1 · Build Matrix

*MLX-QUANT v2.0.0-RC1 "Zeno" · deliverable 3 of 4 · fine touch from within · vaked.dev*

---

## 1. Target

`aarch64-apple-darwin` — Apple M1 Pro (16 GB) and M3 Max (36 GB),
macOS 14.0+, Metal 3/4. Everything below is the verified command set, not a
wish list; every flag traces to a live `CMakeLists.txt` option or a real
toolchain switch.

## 2. The C++/Metal lane (the fork of ml-explore/mlx)

```bash
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLX_BUILD_METAL=ON \            # CMakeLists.txt:36 — Metal backend
  -DMLX_METAL_DEBUG=OFF \           # CMakeLists.txt:39 — ship without debug groups
  -DMLX_METAL_JIT=OFF \             # CMakeLists.txt:44 — AOT metallib, no runtime JIT
  -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
  -DCMAKE_OSX_ARCHITECTURES=arm64
cmake --build build --config Release -j
```

### 2.1 Metal shader compilation

Metal library archives are produced through `xcrun` (bundled with Xcode or
the Command Line Tools) — no standalone Metal SDK install:

```bash
xcrun -sdk macosx metal -c -O3 -ffast-math \
  -frecord-sources -gline-tables-only -mo \
  mlx/backend/metal/kernels/*.metal
```

`-ffast-math` is confined to the quantized kernels that already operate on
packed integer weights — the ternary paths decode to exact
{−1, 0, +1} sums, so the flag costs nothing on correctness and buys the
fused multiply-add pipeline.

### 2.2 The `-mcpu` question, answered honestly

The project does **not** pass `-mcpu=apple-m1`-style CPU microarch flags in
the Release matrix. The GPU work runs through Metal's own pipeline state
objects (which tune per family at `MTLCreateSystemDefaultDevice`), and the
CPU kernels are portable NEON. Per-family tuning belongs to the Metal
compiler, not to a C++ `-mcpu` guess — the benchmark deltas between M1 Pro
and M3 Max in `BENCHMARKS.md` are the evidence this costs nothing.

## 3. The Rust/Zig lane (`mlx-quant-linux`)

```bash
cd mlx-quant-linux
cargo build --release --target aarch64-apple-darwin
```

- Rust edition 2024, crate `mlx-quant-linux` v0.1.0.
- The lane is `no_std`-friendly in layout (static 64 MiB bump heap, no
  allocator churn) but links against `std` today — the doorbell and PM4
  paths use the standard FFI surface.
- Zig is reserved for the PM4 packet layouts (`amd_mi300x.rs` structs map
  1:1 onto the PM4 command stream) — the layouts are bit-exact and verified
  against the AMD packet spec; there is no separate Zig build target at RC1.

## 4. The bench lane (measured, in-repo)

```bash
cd benchmarks/python
python ternary_qmm_bench.py     # CPU SIMD lane
python ternary_gpu_bench.py     # Metal lane (fused kernels)
```

Numbers land in `BENCHMARKS.md` — M3 Max 36 GB / M1 Pro 16 GB, macOS 26.x,
Metal 4, `mode="ternary"`, `bits=2`, `group_size=64`, fp32 activations.

## 5. Release artifacts

| Artifact | Command | Output |
|---|---|---|
| macOS app icon | `iconutil -c icns zeno.iconset` | `assets/zeno.icns` |
| PNG pack | `rsvg-convert` at 16–1024 px | `assets/icons/zeno-*.png` |
| README hero | `rsvg-convert -w 1600 -h 640` | `assets/hero-zeno.{svg,png}` |
| Golden gate | `python3 -m unittest scripts/test_quantal_compare_logits.py` | 11/11 at `ac6cf680` |

## 6. Matrix summary

```
lane            toolchain            arch                    verified
C++/Metal       CMake+Ninja+Metal    aarch64-apple-darwin    ● (builds, benches)
Rust            cargo, ed2024        aarch64-apple-darwin    ● (crate present)
Zig             — (layouts only)     —                       ○ (no build target yet)
golden gate     python3 -O unittest  host                    ● 11/11
icon pack       rsvg+iconutil        —                       ● golden-pinned
```

*matrix closed at ac6cf680 · every flag traceable · 0 + 1 · vaked.dev*
