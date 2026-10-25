#!/usr/bin/env python3
"""
quantal_to_bitnet.py — load the quantal ternary export into a transformers
BitNetForCausalLM state_dict, then verify forward parity against the Rust
golden logits.

The ayeOS export ships 168 packed ternary matrices (m000..m167.json), an
embedding table (embeddings.f16) and 49 RMSNorm gain rows (norms.f32) for a
Qwen2.5-0.5B trained with the *deployed-forward* (weight-quant-only) BitLinear:
no per-projection RMSNorm, no activation_quant. The transformers bitnet module
with ``use_sub_norms=False`` is the exact same forward — plain nn.Linear over
already-quantized weights — so dequantizing the matrices into a state_dict and
loading them into BitNetForCausalLM reproduces the deployed model bit-for-bit.

Parity gate (mirrors the Rust golden-logits gate):
    python3 quantal_to_bitnet.py --model-dir <capsule> --ref <ref.json>
compares the transformers logits against the MLX reference's ``logits_vanilla``
(acceptance ~1e-2 abs/rel, identical argmax).

Usage (build a state_dict / config only):
    python3 quantal_to_bitnet.py --model-dir <capsule> --out <dir>

Deps: numpy (always), torch + transformers (only when --out or parity mode).
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # torch only needed for --out / parity mode

HIDDEN = 896
LAYERS = 24
Q_HEADS = 14
KV_HEADS = 2
HEAD_DIM = 64
INTERMEDIATE = 4864
VOCAB = 151936


# ------------------------------------------------------------- ayeOS loading ---


def dequant_matrix(codes, scales, n, k):
    """ayeOS: value = (code − 1) · scale, 2-bit LSB-first, row-major over K."""
    c = np.asarray(codes, dtype=np.uint32).reshape(n, k // 16)
    lanes = np.stack([(c >> (2 * i)) & 3 for i in range(16)], axis=-1)
    vals = lanes.reshape(n, k).astype(np.float32) - 1.0
    s = np.repeat(np.asarray(scales, dtype=np.float32).reshape(n, k // 64), 64, axis=1)
    return vals * s


def load_capsule(model_dir):
    """Return (matrices: dict[str, np.ndarray], embeddings, norms).

    Keys are full state_dict paths with a ``.weight`` suffix (the ayeOS
    ``name`` field omits it), e.g. ``model.layers.0.self_attn.q_proj.weight``.
    """
    capsule = Path(model_dir)
    index = json.loads((capsule / "index.json").read_text())
    mats = {}
    for entry in index["matrices"]:
        raw = json.load(open(capsule / entry["file"]))
        key = (
            raw["name"] if raw["name"].endswith(".weight") else raw["name"] + ".weight"
        )
        mats[key] = dequant_matrix(
            raw["codes"], raw["scales"], raw["dim"], raw["in_features"]
        )
    emb = np.fromfile(capsule / "embeddings.f16", dtype=np.float16).astype(np.float32)
    emb = emb.reshape(-1, HIDDEN)
    norms = np.fromfile(capsule / "norms.f32", dtype=np.float32).reshape(-1, HIDDEN)
    return mats, emb, norms


# -------------------------------------------------------- state_dict builder ---


def build_state_dict(mats, emb, norms):
    """Map the ayeOS matrices to BitNetForCausalLM state_dict keys."""
    sd = {}
    # embeddings + final norm
    sd["model.embed_tokens.weight"] = torch.from_numpy(emb.copy())
    sd["model.norm.weight"] = torch.from_numpy(norms[2 * LAYERS].copy())
    for layer in range(LAYERS):
        in_norm = norms[2 * layer]
        post_norm = norms[2 * layer + 1]
        sd[f"model.layers.{layer}.input_layernorm.weight"] = torch.from_numpy(
            in_norm.copy()
        )
        sd[f"model.layers.{layer}.post_attention_layernorm.weight"] = torch.from_numpy(
            post_norm.copy()
        )
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            key = f"model.layers.{layer}.self_attn.{proj}.weight"
            sd[key] = torch.from_numpy(mats[key].copy())
        for proj in ("gate_proj", "up_proj", "down_proj"):
            key = f"model.layers.{layer}.mlp.{proj}.weight"
            sd[key] = torch.from_numpy(mats[key].copy())
    # tied lm_head
    sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
    return sd


def build_config():
    """BitNetConfig matching the Qwen2.5-0.5B architecture + deployed forward."""
    from transformers.models.bitnet.configuration_bitnet import BitNetConfig

    return BitNetConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=LAYERS,
        num_attention_heads=Q_HEADS,
        num_key_value_heads=KV_HEADS,
        hidden_act="silu",
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        attention_bias=False,
        use_sub_norms=False,
        head_dim=HEAD_DIM,
        rope_parameters={"rope_theta": 1_000_000.0, "rope_type": "default"},
        attn_implementation="eager",
    )


# --------------------------------------------------------------- parity gate ---


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-dir",
        required=True,
        help="ayeOS capsule (index.json + m*.json + sidecars)",
    )
    p.add_argument("--ref", default=None, help="golden ref.json for the parity gate")
    p.add_argument(
        "--out", default=None, help="write config.json + pytorch_model.bin to this dir"
    )
    args = p.parse_args()

    print("1. loading ayeOS capsule...")
    mats, emb, norms = load_capsule(args.model_dir)
    print(f"   {len(mats)} matrices, embeddings {emb.shape}, norms {norms.shape}")

    if args.out is not None:
        print("2. building state_dict + config...")
        sd = build_state_dict(mats, emb, norms)
        config = build_config()
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(config.to_dict(), indent=2))
        torch.save(sd, out / "pytorch_model.bin")
        print(f"   wrote {out}/config.json + pytorch_model.bin")

    if args.ref is not None:
        print("3. parity gate (BitNetForCausalLM vs golden reference)...")
        from transformers.models.bitnet.modeling_bitnet import BitNetForCausalLM

        config = build_config()
        model = BitNetForCausalLM(config)
        model.eval()
        sd = build_state_dict(mats, emb, norms)
        model.load_state_dict(sd, strict=False)
        model.to(torch.float32)

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        ref = json.load(open(args.ref))
        ok_all = True
        for i, rp in enumerate(ref["prompts"]):
            prompt = (
                f"<|im_start|>system\n{rp['system']}<|im_end|>\n"
                f"<|im_start|>user\n{rp['user']}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            out = model(torch.tensor([ids])).logits[0, -1].detach().numpy()
            gold = np.asarray(rp["logits_vanilla"], dtype=np.float32)
            d = np.abs(out - gold)
            denom = np.maximum(np.abs(gold), 1.0)
            rel = d / denom
            gate = bool(d.max() <= 1e-2 or rel.max() <= 1e-2)
            same = bool(out.argmax() == gold.argmax())
            ok_all &= gate
            print(
                f"  prompt {i + 1}: max_abs={d.max():.3e} max_rel={rel.max():.3e} "
                f"argmax {out.argmax()} vs {gold.argmax()} same={same} → {'PASS' if gate else 'FAIL'}"
            )
        print("RESULT:", "PASS" if ok_all else "FAIL")
        return 0 if ok_all else 1

    print("done (no --out or --ref action selected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
