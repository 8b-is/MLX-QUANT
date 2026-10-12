# RC1 · Capability Graph Mapping

*MLX-QUANT v2.0.0-RC1 "Zeno" · deliverable 2 of 4 · fine touch from within · vaked.dev*

---

## 1. The node: Zeno over the Vaked capability graph

```
                ┌────────────────────────────────────────────────┐
                │  Zeno wallet layer  (request, signed)           │
                │  triggers local inference, never sees weights   │
                └──────────────────┬─────────────────────────────┘
                                   │ capability: INFER::trigger
                                   ▼
   ┌───────────────────────────────────────────────────────────────┐
   │  Vaked capability node  —  mem8 ledger                         │
   │  · admit: hash-verify request against ledger                   │
   │  · route: CrabCC ring → UMA (zero-copy, no OS crossing)        │
   │  · dispatch: polar queue doorbell (O(1))                       │
   │  · record: state hash → mem8, append-only                     │
   └──────┬───────────────────────────────┬────────────────────────┘
          │ capability: WEIGHTS::read     │ capability: KERNEL::dispatch
          ▼                               ▼
   ┌─────────────────┐            ┌──────────────────────────────┐
   │  weights (2/3/4b)│            │  Metal queue · fused ternary │
   │  read-only map  │            │  kernels {−1, 0, +1}         │
   └─────────────────┘            └──────────────────────────────┘
```

**The rule the graph encodes:** the wallet can *trigger*, the ledger can
*route and record*, the kernels can *compute* — and no edge grants any node
more than it needs. The wallet never reads weights; the kernel never sees the
request; the ledger is the only writer of state.

## 2. Vaked node configuration (v1, RC1)

```toml
[node.zeno-wallet]
layer = "wallet"
grants = ["INFER::trigger"]

[node.mem8]
layer = "ledger"
grants = ["REQUEST::admit", "STATE::record"]
verify = "sha256"          # state hash-verified, append-only
entry = "mem8:/mlx-quant/rc1"

[node.crabcc-ring]
layer = "transport"
grants = ["RING::feed"]
mode = "zero-allocation DMA"
latency_budget = "sub-millisecond"

[node.weights]
layer = "params"
grants = ["WEIGHTS::read"]
format = "ternary packed {−1,0,+1} · 12.80x"

[node.polar-queue]
layer = "dispatch"
grants = ["KERNEL::dispatch"]
bypass = "O(1) doorbell"
```

## 3. The isolation truth, per platform

The directive names eBPF and seccomp BPF filters. Those are Linux
facilities. The honest mapping at RC1:

| Isolation | Linux lane (`mlx-quant-linux`) | macOS lane (Metal backend) |
|---|---|---|
| eBPF filter | available — `cgroup`/`seccomp` attach point | **does not exist** |
| seccomp BPF | available | **does not exist** |
| Equivalent on macOS | — | `sandbox-exec` / Seatbelt profiles + MACF; enforced by the host agent, not the kernel module |
| Capability graph | enforced in software at the node router | enforced in software at the node router |

**RC1 verdict: the capability graph is enforced in userspace on both
platforms.** The Linux lane can additionally drop into real eBPF/seccomp
when it runs on a Linux host; the macOS lane runs the same graph with the
same grants, mediated by the agent rather than the kernel. No release claim
of "eBPF isolation" is made for darwin — it is not possible, and the RC1
notes say so.

## 4. The trigger path, step by step

1. Zeno wallet signs a trigger (`INFER::trigger`, request id, prompt hash).
2. mem8 admits it: hash-verify against the ledger, then routes.
3. CrabCC feeds context into Unified Memory over the DMA ring — no copy, no
   `memcpy`, no kernel crossing.
4. The polar queue dispatches via doorbell; the fused ternary kernels run.
5. mem8 records the state hash; the wallet receives the response id, not the
   weights, not the intermediates.

Each transition is a hash-verified edge. That is the capability graph: five
nodes, four edges, no edge the wallet can use to read a weight.

*graph closed at ac6cf680 · platform truth labeled · 0 + 1 · vaked.dev*
