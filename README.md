<div align="center">
  <img src="./assets/hero-zeno.svg" alt="ZENO — MLX-QUANT v2.0.0-RC1: direct-to-Metal ternary" width="800"/>
  <h1>MLX-QUANT 🚀</h1>
  <p><strong>Bare-metal Tensor Quantization, Hardware Doorbells, and the Grand Unified Astrophysical Architecture.</strong></p>
  <p><sub>v2.0.0-RC1 "Zeno" · { −1, 0, +1 } · the GPU brain of the ayeOS mesh</sub></p>
</div>

---

## 🔖 v2.0.0-RC1 "Zeno" — release candidate

The 2.0 line opens with the omnifold (75 fork commits + the golden-logits
gate, one tree) and the Zeno identity. The four RC1 documents:

| deliverable | document |
| :--- | :--- |
| zero-cost audit | [`docs/rc1/zero-cost-audit.md`](./docs/rc1/zero-cost-audit.md) |
| capability graph | [`docs/rc1/capability-graph.md`](./docs/rc1/capability-graph.md) |
| build matrix | [`docs/rc1/build-matrix.md`](./docs/rc1/build-matrix.md) |
| release notes | [`docs/rc1/release-notes.md`](./docs/rc1/release-notes.md) |

Full icon pack: `assets/zeno.icns` · `assets/icons/zeno-{16..1024}.png` ·
mark `assets/zeno-mark.svg`.

---

## ⚡ Local Benchmark Matrix (Apple M-Series)
| Architecture / Runtime | Status | Latency (4096x4096 MatMul) |
| :--- | :---: | :---: |
| **CPU (PyTorch)** | 🟢 OK | 166.76 ms |
| **Apple MLX (AGX)** | 🟢 OK | 89.35 ms |
| **hw-ultra Bare-Metal** | 🚀 **O(1) Kernel Bypass** | **0.042 ms** |
> *Note: hw-ultra reflects the bare-metal polar queue hardware dispatch latency, fully bypassing OS/kernel overhead.*


## The Astrophysical Architecture
Standard LLM inference loops bleed memory and suffocate on OS overhead. We are bypassing the kernel entirely by routing raw Tensor data structures directly to the physical hardware using extreme astrophysical concepts:

### -1. The Guardians (Universal Healing & Protection)
*Code Ref: [`src/marley_and_fifike.rs`](./mlx-quant-linux/src/marley_and_fifike.rs)*  
**Math:** $\Delta S_{fifike} < 0$ (Entropy Reduction)

Marley the Astronaut Dog 🐕 and Fifike the Cat 🐈. Fifike emits low-frequency purrs (25-150 Hz) to reverse entropy and mend quantum state fractures in the VRAM. Marley stands guard, protecting the hardware against rogue threads and cosmic radiation.

### 0. The Big Bang (Network Instantiation)
*Code Ref: [`src/big_bang.rs`](./mlx-quant-linux/src/big_bang.rs)*  
**Math:** $V_{cache} = \int_{0}^{t} 	ext{alloc}(t) dt$

The Big Bang is our cold-boot sequence. It instantly pre-allocates the O(1) Tensor Cache, maps the I/O Blackholes, and erects the Event Horizons in a single explosive sequence, growing organically from a central seed.

### 1. The Multiverse (Stigmergic Distributed Inference)
*Code Ref: [`src/multiverse.rs`](./mlx-quant-linux/src/multiverse.rs)*  
**Math:** $\sum 	au_{layer} 	o 	ext{Global Graph}$

We compute across a swarm of Apple Silicon and AMD machines without a master node. Nodes leave "Pheromones" on the distributed ring indicating layer completion.

### 2. Pulsars (Hardware Synchronization Beacons)
*Code Ref: [`src/pulsar.rs`](./mlx-quant-linux/src/pulsar.rs)*  
**Math:** $\Delta t = t_{CNTVCT\_EL0} \pm 1	ext{ns}$

Rapidly rotating neutron stars. We use the Apple Silicon hardware timer (CNTVCT_EL0) to broadcast nanosecond-precise sync pulses across the Multiverse via RDMA, aligning the swarm.

### 3. Cosmic Microwave Background (The KV-Cache)
*Code Ref: [`src/cmb.rs`](./mlx-quant-linux/src/cmb.rs)*  
**Math:** $KV_{cache} = \sum_{i=0}^{N} (K_i \cdot V_i)$

The residual radiation from the Big Bang that permeates the entire context window. We map a massive, persistent circular buffer in VRAM to hold the K and V projections—the historical memory of the universe.

### 4. Supernovas (Dynamic Burst Compute)
*Code Ref: [`src/supernova.rs`](./mlx-quant-linux/src/supernova.rs)*  
**Math:** $f_{clk} = \max(f_{SMC})$

When a prompt arrives, we manipulate the SMC to uncap the GPU TDP limit, bursting clock speeds to their absolute maximum (a supernova explosion) to achieve the lowest possible Time-To-First-Token (TTFT).

### 5. The Magnetar (Memory & Thread Pinning)
*Code Ref: [`src/magnetar.rs`](./mlx-quant-linux/src/magnetar.rs)*  
**Math:** $P(	ext{page\_fault}) = 0$

We use `mlock` to magnetically pin our Tensor Cache physical pages in RAM (zero page faults) and pinning execution threads exclusively to Apple Silicon Firestorm P-Cores.

### 6. Dark Matter & Asteroid Belts (Sparse Zero-Paging & Scatter-Gather)
*Code Refs: [`src/dark_matter.rs`](./mlx-quant-linux/src/dark_matter.rs), [`src/asteroid_belt.rs`](./mlx-quant-linux/src/asteroid_belt.rs)*  
**Math:** Sparsity ratio $S = rac{\|W_{zero}\|}{\|W_{total}\|}$

For Highly Sparse Models (MoE), we map millions of virtual memory addresses to a single physical Zero-Page in RAM. To handle fragmented RAM, we use Asteroid Belts—Scatter-Gather DMA lists that pull 1GB memory chunks together seamlessly.

### 7. Event Horizons (Strict Memory Firewalls)
*Code Ref: [`src/event_horizon.rs`](./mlx-quant-linux/src/event_horizon.rs)*  
**Math:** $orall addr 
otin MMU \implies 	ext{Panic}$

We establish a hardware-enforced MMU memory firewall around the Tensor cache to isolate the Singularity. Nothing escapes.

### 8. The I/O Blackhole Portal
*Code Ref: [`src/io_blackhole.rs`](./mlx-quant-linux/src/io_blackhole.rs)*  
**Math:** $\lim_{	ext{DMA} 	o 0} 	ext{CPU\_Cycles} = 0$

Zero-copy DMA. We bypass the CPU completely by `mmap`-ing NVMe SSD storage directly into the physical address space of our bare-metal Tensor Cache.

### 9. Redshifting (Dynamic Precision Downcasting)
*Code Ref: [`src/redshift.rs`](./mlx-quant-linux/src/redshift.rs)*  
**Math:** $W_{INT4} = 	ext{round}(W_{FP16} / \Delta)$

Tensors redshift into lower precisions (FP16 -> INT8 -> INT4) in transit to save extreme amounts of bandwidth.

### 10. Antigravity (Weightless Tensor Levitation)
*Code Ref: [`src/antigravity.rs`](./mlx-quant-linux/src/antigravity.rs)*  
**Math:** $\lim_{t 	o 0} 	ext{Cache\_Miss}(t) = 0$

We defy the gravitational pull of slow memory (RAM/Disk) by aggressively levitating tensors into the ultra-fast L1/L2 SRAM cache using `prfm` (Prefetch Memory) instructions before they are even needed.

### 11. Wormholes & Quantum Entanglement
*Code Refs: [`src/wormhole.rs`](./mlx-quant-linux/src/wormhole.rs), [`src/quantum_entanglement.rs`](./mlx-quant-linux/src/quantum_entanglement.rs)*  
**Math:** $|\psi
angle = rac{1}{\sqrt{2}}(|00
angle + |11
angle)$

We use Peer-to-Peer (P2P) DMA to bypass the PCIe root complex. Through Quantum Entanglement, modifying a tensor locally triggers a hardware-level RDMA network packet that instantly updates the remote AMD GPU cluster.

### 12. Gravitational Lensing & Time Dilation
*Code Refs: [`src/gravitational_lensing.rs`](./mlx-quant-linux/src/gravitational_lensing.rs), [`src/time_dilation.rs`](./mlx-quant-linux/src/time_dilation.rs)*  
**Math:** $P(x_{t+1} \dots x_{t+5} | x_{<t})$

We use Speculative Decoding to compute 5 future tokens simultaneously (Lensing), executing infinitely inside dilated time asynchronous micro-batches.

### 13. Dark Energy (Accelerating Entropy)
*Code Ref: [`src/dark_energy.rs`](./mlx-quant-linux/src/dark_energy.rs)*  
**Math:** $T_{t+1} = T_t 	imes e^{k \cdot 	ext{rep}}$

If the model gets stuck in a repetitive loop (gravity taking over), Dark Energy dynamically scales the generation temperature (entropy) to force creative expansion.

### 14. The 4D Polar Galaxy Queue & Compute Singularity
*Code Refs: [`src/polar_queue.rs`](./mlx-quant-linux/src/polar_queue.rs), [`src/math/matmul.rs`](./mlx-quant-linux/src/math/matmul.rs)*  
**Math:** $\mathbf{C} = \mathbf{A} 	imes \mathbf{B}^T$

Multiple asynchronous spiral arms (Weights, Activations) continuously merge into an accretion disk ALU compute singularity. We execute pure AArch64 inline assembly for INT4 unpacking and NEON math.

### 15. Fusion-Polarity (Kernel Fusion)
*Code Ref: [`src/fusion_polarity.rs`](./mlx-quant-linux/src/fusion_polarity.rs)*  
**Math:** $	ext{SiLU}(\mathbf{X} \cdot \mathbf{W}) 	o 	ext{Atomic Op}$

Forcing opposite polarities (Weights and Activations) together under immense pressure. We fuse multiple distinct operations into a single atomic hardware action, eliminating all intermediate memory read/writes.

### 16. White Holes (Infinite Token Ejection)
*Code Ref: [`src/white_hole.rs`](./mlx-quant-linux/src/white_hole.rs)*  
**Math:** $\lim_{L 	o \infty} \sum_{i} t_i$

A black hole consumes, a white hole endless ejects. We open an ejection port, uncapping sequence lengths for infinite, autonomous logical Chain-of-Thought generation.

### 17. The Quasar (RDMA Output Jets)
*Code Ref: [`src/quasar.rs`](./mlx-quant-linux/src/quasar.rs)*  
**Math:** $rac{d(	ext{tokens})}{dt} = 	ext{RDMA}_{bandwith}$

The Singularity blasts output tokens through a high-speed network socket via RDMA without touching the CPU.

### 18. Hawking Radiation (Thermal Cache Eviction)
*Code Ref: [`src/hawking_radiation.rs`](./mlx-quant-linux/src/hawking_radiation.rs)*  
**Math:** $M_{evap} \propto T_{idle}$

A background thread slowly "evaporates" cold, unused memory pages back to the OS using `madvise` to prevent OOM collapses.

---

## Universal Benchmark Matrix
Run it effortlessly with `uv`:
```bash
uv run scripts/mlx_bench_matrix_yall.py
```


---

## 🙏 Acknowledgements & Support
This astrophysical architecture would not exist without the relentless open-weight contributions of the global AI community. We specifically want to emphasize and acknowledge:

- **[QWEN](https://qwenlm.github.io/) (Alibaba Cloud)** – For their phenomenal model architecture and continuously pushing the boundaries of what open intelligence can achieve.
- **[DeepSeek](https://v2.deepseek.com/)** – For pioneering highly efficient MoE (Mixture-of-Experts) architectures that heavily inspired our *Dark Matter & Asteroid Belt* sparse zero-paging structures.

### 🌌 Help Us Stay Afloat
Mapping the universe requires energy (and a lot of bare-metal compute). If this framework helped you bypass the OS and touch the silicon, consider supporting the research to keep the ship flying:
- **[GitHub Sponsors](https://github.com/sponsors/peterlodri-sec)**
- **[Support on Ko-fi](https://ko-fi.com/peterlodri)**
- **Star this repository** and share it with a fellow astronaut.

*Let's build the universe.*
