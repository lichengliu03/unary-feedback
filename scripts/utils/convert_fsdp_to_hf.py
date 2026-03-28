#!/usr/bin/env python3
"""Convert FSDP sharded checkpoint to HuggingFace format."""

import sys
import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import shutil

def convert_fsdp_to_hf(checkpoint_dir: str, output_dir: str, base_model: str, world_size: int = 2):
    """Convert FSDP sharded checkpoint to HuggingFace format.

    Args:
        checkpoint_dir: Path to checkpoint directory containing model_world_size_*_rank_*.pt files
        output_dir: Path to save HuggingFace format model
        base_model: Base model name or path
        world_size: Number of shards (default: 2)
    """
    checkpoint_path = Path(checkpoint_dir)
    output_path = Path(output_dir)

    # Check if checkpoint files exist
    shard_files = [checkpoint_path / f"model_world_size_{world_size}_rank_{rank}.pt"
                   for rank in range(world_size)]

    missing_files = [f for f in shard_files if not f.exists()]
    if missing_files:
        print(f"Error: Missing checkpoint files: {missing_files}")
        return False

    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True
    )

    print(f"Loading sharded checkpoints from: {checkpoint_path}")
    # Load all shards
    sharded_state_dicts = []
    for rank in range(world_size):
        shard_file = checkpoint_path / f"model_world_size_{world_size}_rank_{rank}.pt"
        print(f"  Loading shard {rank}/{world_size}: {shard_file}")
        shard_state_dict = torch.load(shard_file, map_location="cpu", weights_only=False)
        sharded_state_dicts.append(shard_state_dict)

    print("Merging sharded state dicts...")
    # Merge sharded state dicts
    # FSDP sharded state dict has keys like: "_fsdp_wrapped_module.model.layers.0.self_attn.q_proj.weight"
    # We need to merge the sharded tensors
    merged_state_dict = {}

    # Collect all unique keys
    all_keys = set()
    for shard in sharded_state_dicts:
        all_keys.update(shard.keys())

    for key in all_keys:
        # Get tensors from all shards that have this key
        tensors = [shard[key] for shard in sharded_state_dicts if key in shard]

        if len(tensors) == 1:
            # Only one shard has this key, use it directly
            tensor = tensors[0]
            # Convert DTensor to regular tensor if needed
            if hasattr(tensor, '_local_tensor'):
                tensor = tensor._local_tensor
            merged_state_dict[key] = tensor
        else:
            # Multiple shards have this key, need to merge
            # Convert DTensors to regular tensors first
            local_tensors = []
            for t in tensors:
                if hasattr(t, '_local_tensor'):
                    local_tensors.append(t._local_tensor)
                else:
                    local_tensors.append(t)

            # For FSDP, sharded tensors are typically concatenated along dim 0
            merged_state_dict[key] = torch.cat(local_tensors, dim=0)

    # Remove FSDP wrapper prefix if present
    cleaned_state_dict = {}
    for key, value in merged_state_dict.items():
        # Remove "_fsdp_wrapped_module." prefix
        clean_key = key.replace("_fsdp_wrapped_module.", "")
        cleaned_state_dict[clean_key] = value

    print("Loading merged weights into model...")
    # Load the merged state dict into the model
    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)

    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")

    print(f"Saving model to: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))

    # Copy tokenizer
    print("Copying tokenizer...")
    # First try to copy from checkpoint's huggingface dir
    # IMPORTANT: Check if output_path is the same as hf_dir to avoid loading from incomplete directory
    hf_dir = checkpoint_path / "huggingface"

    # If output_path == hf_dir, the directory is being created now, so look for backup or use base model
    if output_path.resolve() == hf_dir.resolve():
        print("Output path is the same as hf_dir, looking for backup or using base model...")
        # Try to find backup directory
        hf_backup = checkpoint_path / "huggingface.old"
        if hf_backup.exists():
            print(f"Found backup tokenizer at: {hf_backup}")
            tokenizer = AutoTokenizer.from_pretrained(str(hf_backup), trust_remote_code=True)
            tokenizer.save_pretrained(str(output_path))
            # Copy chat template if exists
            chat_template_src = hf_backup / "chat_template.jinja"
            if chat_template_src.exists():
                chat_template_dst = output_path / "chat_template.jinja"
                shutil.copy(chat_template_src, chat_template_dst)
                print("Copied chat template from backup")
        else:
            print("No backup found, using base model tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            tokenizer.save_pretrained(str(output_path))
    elif hf_dir.exists():
        print(f"Copying tokenizer from: {hf_dir}")
        tokenizer = AutoTokenizer.from_pretrained(str(hf_dir), trust_remote_code=True)
        tokenizer.save_pretrained(str(output_path))

        # Copy chat template if exists
        chat_template_src = hf_dir / "chat_template.jinja"
        if chat_template_src.exists():
            chat_template_dst = output_path / "chat_template.jinja"
            shutil.copy(chat_template_src, chat_template_dst)
            print("Copied chat template")
    else:
        # Fall back to base model tokenizer
        print("No existing tokenizer found, using base model tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.save_pretrained(str(output_path))

    print("Done!")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python convert_fsdp_to_hf.py <checkpoint_dir> <output_dir> <base_model> [world_size]")
        print("Example: python convert_fsdp_to_hf.py /path/to/ckpt/global_step_50/actor /path/to/output Qwen/Qwen2.5-3B-Instruct 2")
        sys.exit(1)

    checkpoint_dir = sys.argv[1]
    output_dir = sys.argv[2]
    base_model = sys.argv[3]
    world_size = int(sys.argv[4]) if len(sys.argv) > 4 else 2

    success = convert_fsdp_to_hf(checkpoint_dir, output_dir, base_model, world_size)
    sys.exit(0 if success else 1)
