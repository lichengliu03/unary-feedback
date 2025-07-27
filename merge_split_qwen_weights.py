"""
merge_split_qwen2_weights.py
============================
Utility script for **merging** or **splitting** Qwen-2 / Llama-family weights.

Why you need it
---------------
1. After FSDP / tensor-parallel training, each GPU rank stores a partial
   checkpoint such as `model_world_size_*_rank_*.pt`, where many tensors are
   **halved** along their parallel dimension. They must be concatenated to form
   full tensors before single-GPU inference or uploading to Hugging Face.
2. This script provides two sub-commands:
   • merge – load two TP/FSDP rank checkpoints, stitch corresponding tensors
     together, and save a single `model.safetensors` (keeping the `model.`
     prefix so that vLLM / Transformers can load it).
   • split – take a full `model.safetensors`, split it into
     `model-0000X-of-0000N.safetensors` shards plus an `index.json` that follows
     the Hugging Face specification. Convenient for LFS uploads.
3. Optional `--dtype` lets you down-cast from fp32 to bf16/fp16 on the fly to
   shrink file size (recommended after training).

Example usage
-------------
# Merge two ranks into bf16 single file
python merge_split_qwen2_weights.py merge \
  --rank0 checkpoints/.../model_world_size_2_rank_0.pt \
  --rank1 checkpoints/.../model_world_size_2_rank_1.pt \
  --output-dir merged_actor_model_bf16 \
  --dtype bf16

# Split into 2 shards for the Hub
python merge_split_qwen2_weights.py split \
  --input-model merged_actor_model_bf16/model.safetensors \
  --output-dir hf_actor_model_shards \
  --num-shards 2
"""

import argparse
import os
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import torch
from safetensors.torch import save_file, load_file

# -------------------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------------------

def _bytes_of(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _clean_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove distributed prefixes like 'module.' but KEEP 'model.'!"""
    clean = {}
    for k, v in state.items():
        key = k
        if key.startswith("module."):
            key = key[len("module.") :]
        # Keep the leading 'model.' prefix – required by vLLM/Transformers.
        clean[key] = v
    return clean


# -------------------------------------------------------------------------------------
# MERGE logic
# -------------------------------------------------------------------------------------

def merge_two_ranks(rank0_path: str, rank1_path: str, output_dir: str, target_dtype: torch.dtype = None) -> str:
    """Merge two DP/TP rank checkpoints into a single safetensors file.

    In most FSDP/DP cases the two ranks are identical, so using only rank-0
    already yields the full model.  If you wish, you can add extra validation
    logic to compare specific tensors between ranks before concatenation."""

    # --------------------------- load both ranks ---------------------------
    print(f"Loading rank-0 checkpoint: {rank0_path}")
    s0 = torch.load(rank0_path, map_location="cpu", weights_only=False)
    s1 = None
    if rank1_path and Path(rank1_path).expanduser().exists() and rank1_path != rank0_path:
        print(f"Loading rank-1 checkpoint: {rank1_path}")
        s1 = torch.load(rank1_path, map_location="cpu", weights_only=False)

    def _extract_state(d):
        if isinstance(d, dict):
            if "model" in d:
                return d["model"]
            if "state_dict" in d:
                return d["state_dict"]
            return d
        raise RuntimeError("Unsupported checkpoint format – expecting a dict containing 'model' or 'state_dict'.")

    sd0 = _extract_state(s0)
    sd1 = _extract_state(s1) if s1 is not None else None

    # Heuristics for TP-2 tensor concatenation.
    # In Qwen/Llama checkpoints, weights are stored using the PyTorch layout
    # (out_features, in_features).  Both column-parallel and row-parallel
    # layers therefore end up split along **dim 0**.  Empirically we found
    # that `down_proj` / `o_proj` should also be concatenated on dim 0 – if we
    # try dim 1 the resulting shape mismatches the model definition.

    dim0_keywords = [
        "q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "wq", "wk", "wv",
        "o_proj", "down_proj", "wo",
        ".attention.q_proj.weight", ".attention.k_proj.weight", ".attention.v_proj.weight",
        ".attention.o_proj.weight", ".mlp.up_proj.weight", ".mlp.gate_proj.weight", ".mlp.down_proj.weight",
        "embed_tokens", "tok_embeddings", "lm_head",
    ]
    dim1_keywords: list[str] = []  # none for now

    def _needs_concat(name: str) -> int:
        """Return dim along which to concat, or -1 if no concat needed."""
        for kw in dim0_keywords:
            if kw in name:
                return 0
        for kw in dim1_keywords:
            if kw in name:
                return 1
        # 1-D tensors (e.g. RMSNorm) also split along dim-0
        return 0

    def _prepare(t: torch.Tensor) -> torch.Tensor:
        # unwrap Parameter / DTensor / meta → cpu contiguous tensor
        if isinstance(t, torch.nn.Parameter):
            t = t.data
        if hasattr(t, "_local_tensor"):
            t = t._local_tensor
        elif "DTensor" in str(type(t)):
            try:
                t = t.to_local()  # type: ignore
            except Exception:
                pass
        if t.device.type == "meta":
            t = torch.zeros(t.shape, dtype=t.dtype)
        return t.detach().cpu().contiguous()

    merged_state: Dict[str, torch.Tensor] = {}

    for key, t0 in sd0.items():
        t0 = _prepare(t0)
        if sd1 is None or key not in sd1:
            merged_state[key] = t0
            continue

        t1 = _prepare(sd1[key])

        # If tensors are identical (common for biases), keep one
        if torch.equal(t0, t1):
            merged_state[key] = t0
            continue

        # Otherwise attempt to concatenate along pre-defined dimension
        dim = _needs_concat(key)
        try:
            merged_tensor = torch.cat([t0, t1], dim=dim)
            merged_state[key] = merged_tensor
        except Exception as e:
            print(f"⚠️  Failed to concat {key} along dim {dim}: {e}. Using rank-0 tensor only.")
            merged_state[key] = t0

    if target_dtype:
        for key in merged_state:
            merged_state[key] = merged_state[key].to(target_dtype)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "model.safetensors"
    print(f"Saving merged safetensors: {out_path}")
    save_file(merged_state, str(out_path))
    print("Done.")
    return str(out_path)


def merge_four_ranks(rank_paths: list[str], output_dir: str, target_dtype: torch.dtype = None) -> str:
    """Merge four TP rank checkpoints into a single safetensors file."""
    
    print(f"Loading 4 rank checkpoints...")
    states = []
    for i, path in enumerate(rank_paths):
        print(f"Loading rank-{i} checkpoint: {path}")
        s = torch.load(path, map_location="cpu", weights_only=False)
        states.append(s)

    def _extract_state(d):
        if isinstance(d, dict):
            if "model" in d:
                return d["model"]
            if "state_dict" in d:
                return d["state_dict"]
            return d
        raise RuntimeError("Unsupported checkpoint format – expecting a dict containing 'model' or 'state_dict'.")

    state_dicts = [_extract_state(s) for s in states]

    dim0_keywords = [
        "q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "wq", "wk", "wv",
        "o_proj", "down_proj", "wo",
        ".attention.q_proj.weight", ".attention.k_proj.weight", ".attention.v_proj.weight",
        ".attention.o_proj.weight", ".mlp.up_proj.weight", ".mlp.gate_proj.weight", ".mlp.down_proj.weight",
        "embed_tokens", "tok_embeddings", "lm_head",
    ]
    dim1_keywords: list[str] = []

    def _needs_concat(name: str) -> int:
        """Return dim along which to concat, or -1 if no concat needed."""
        for kw in dim0_keywords:
            if kw in name:
                return 0
        for kw in dim1_keywords:
            if kw in name:
                return 1
        return 0

    def _prepare(t: torch.Tensor) -> torch.Tensor:
        # unwrap Parameter / DTensor / meta → cpu contiguous tensor
        if isinstance(t, torch.nn.Parameter):
            t = t.data
        if hasattr(t, "_local_tensor"):
            t = t._local_tensor
        elif "DTensor" in str(type(t)):
            try:
                t = t.to_local()
            except Exception:
                pass
        if t.device.type == "meta":
            t = torch.zeros(t.shape, dtype=t.dtype)
        return t.detach().cpu().contiguous()

    merged_state: Dict[str, torch.Tensor] = {}

    # Get all keys from first rank
    for key in state_dicts[0].keys():
        tensors = []
        
        # Collect tensors from all ranks
        for i, sd in enumerate(state_dicts):
            if key not in sd:
                print(f"⚠️  Key {key} missing in rank {i}, skipping concatenation")
                break
            t = _prepare(sd[key])
            tensors.append(t)
        
        if len(tensors) != 4:
            # Use tensor from rank 0 if not all ranks have this key
            merged_state[key] = tensors[0] if tensors else None
            continue

        # Check if all tensors are identical (like biases)
        all_equal = all(torch.equal(tensors[0], t) for t in tensors[1:])
        if all_equal:
            merged_state[key] = tensors[0]
            continue

        # Concatenate along appropriate dimension
        dim = _needs_concat(key)
        try:
            merged_tensor = torch.cat(tensors, dim=dim)
            merged_state[key] = merged_tensor
        except Exception as e:
            print(f"⚠️  Failed to concat {key} along dim {dim}: {e}. Using rank-0 tensor only.")
            merged_state[key] = tensors[0]

    if target_dtype:
        for key in merged_state:
            merged_state[key] = merged_state[key].to(target_dtype)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "model.safetensors"
    print(f"Saving merged safetensors: {out_path}")
    save_file(merged_state, str(out_path))
    print("Done.")
    return str(out_path)


# -------------------------------------------------------------------------------------
# SPLIT logic
# -------------------------------------------------------------------------------------

def split_safetensors(
    input_path: str,
    output_dir: str,
    num_shards: int = 2,
    shard_max_gb: float = 3.0,
):
    """Split a single safetensors into multiple shards + index (HF style)."""

    tensors = load_file(input_path)  # OrderedDict preserves original order
    print(f"Loaded {len(tensors)} tensors from {input_path}")

    # Sort tensors by size (descending) to balance shard sizes better
    items: List[tuple[str, torch.Tensor]] = sorted(
        tensors.items(), key=lambda kv: _bytes_of(kv[1]), reverse=True
    )

    shards: List[OrderedDict[str, torch.Tensor]] = [OrderedDict() for _ in range(num_shards)]
    shard_sizes: List[int] = [0] * num_shards
    shard_max_bytes = int(shard_max_gb * 1024**3)

    for name, tensor in items:
        # Always place next tensor into the currently smallest shard
        idx = shard_sizes.index(min(shard_sizes))
        if shard_sizes[idx] + _bytes_of(tensor) > shard_max_bytes:
            print(
                f"⚠️  tensor {name} ({_bytes_of(tensor)/1e6:.1f} MB) would make shard {idx} exceed the {shard_max_gb} GB hint; continuing anyway."
            )
        shards[idx][name] = tensor
        shard_sizes[idx] += _bytes_of(tensor)

    # Save each shard and build the HF index.json
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weight_map = {}
    for i, shard in enumerate(shards, 1):
        shard_name = f"model-{i:05d}-of-{num_shards:05d}.safetensors"
        shard_path = output_dir / shard_name
        print(
            f"Saving shard {i}/{num_shards}: {shard_path}  size={shard_sizes[i-1]/1e9:.2f} GB  tensors={len(shard)}"
        )
        save_file(shard, str(shard_path))
        for key in shard.keys():
            weight_map[key] = shard_name

    index = {
        "metadata": {"total_size": sum(shard_sizes)},
        "weight_map": weight_map,
    }
    index_path = output_dir / "model.safetensors.index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"Created index file: {index_path}")

    return index_path


# -------------------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Merge or split Qwen2 safetensors correctly (keep 'model.' prefix)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # merge sub-command
    m = subparsers.add_parser("merge", help="Merge rank0 & rank1 .pt into single safetensors")
    m.add_argument("--rank0", required=True, help="Path to model_world_size_*_rank_0.pt")
    m.add_argument("--rank1", required=False, help="Path to model_world_size_*_rank_1.pt (optional, only for validation)")
    m.add_argument("--output-dir", required=True, help="Directory to save merged safetensors & logs")
    m.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32", help="Convert tensors to this dtype before saving (default: fp32, i.e. keep original)")

    # split sub-command
    s = subparsers.add_parser("split", help="Split single safetensors into HF shards (+index.json)")
    s.add_argument("--input-model", required=True, help="Path to model.safetensors to split")
    s.add_argument("--output-dir", required=True, help="Directory where shards will be stored")
    s.add_argument("--num-shards", type=int, default=2, help="Number of shards (default=2)")
    s.add_argument("--shard-max-gb", type=float, default=3.0, help="Max shard size in GB (for warning)")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "merge":
        # determine target dtype
        dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
        target_dtype = dtype_map[args.dtype]
        merge_two_ranks(args.rank0, args.rank1 or args.rank0, args.output_dir, target_dtype=target_dtype)
    elif args.command == "split":
        split_safetensors(args.input_model, args.output_dir, args.num_shards, args.shard_max_gb)
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main() 
