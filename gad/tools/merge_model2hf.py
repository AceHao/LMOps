# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Merge FSDP checkpoints to HuggingFace format.

Usage:
    # Merge FSDP checkpoint for resume training (keeps PEFT structure + lora_adapter/)
    python merge_model2hf.py --local_dir /path/to/checkpoint/actor

    # Clean PEFT prefixes but keep lora_adapter/ separate (for resume with vLLM TP>1)
    python merge_model2hf.py --local_dir /path/to/checkpoint/actor --clean-model-prefix

    # Merge with LoRA baked into weights (for inference/upload)
    python merge_model2hf.py --local_dir /path/to/checkpoint/actor --lora

    # Merge with LoRA and upload to HuggingFace
    python merge_model2hf.py --local_dir /path/to/checkpoint/actor --lora --hf_upload_path username/model --private
"""

from typing import List, Tuple, Dict
import re
import os
import shutil
import torch
import argparse
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification, AutoModelForVision2Seq
from concurrent.futures import ThreadPoolExecutor

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from torch.distributed._tensor import Shard, Placement


def merge_by_placement(tensors: List[torch.Tensor], placement: Placement):
    if placement.is_replicate():
        return tensors[0]
    elif placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    elif placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    else:
        raise ValueError(f"Unsupported placement: {placement}")


def has_lora_adapter(local_dir: str) -> bool:
    """Check if checkpoint has a lora_adapter folder."""
    lora_path = os.path.join(local_dir, "lora_adapter", "adapter_config.json")
    return os.path.exists(lora_path)


def clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove PEFT wrapper prefixes from state dict keys."""
    cleaned = {}
    for key, value in state_dict.items():
        # Skip LoRA weights - they will be merged separately
        if "lora_A" in key or "lora_B" in key:
            continue

        new_key = key
        # Remove PEFT wrapper prefix: base_model.model.X -> X
        if new_key.startswith("base_model.model."):
            new_key = new_key[len("base_model.model."):]
        # Handle base_layer wrapper: X.base_layer.weight -> X.weight
        new_key = new_key.replace(".base_layer.weight", ".weight")
        new_key = new_key.replace(".base_layer.bias", ".bias")

        cleaned[new_key] = value
    return cleaned


def merge_fsdp_shards(local_dir: str) -> Dict[str, torch.Tensor]:
    """Merge FSDP sharded checkpoint into a single state dict."""
    # Find world size
    world_size = 0
    for filename in os.listdir(local_dir):
        match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if match:
            world_size = int(match.group(1))
            break
    assert world_size, "No model file with the proper format"

    print(f"Loading FSDP checkpoint with world_size={world_size}")

    state_dict = torch.load(os.path.join(local_dir, f'model_world_size_{world_size}_rank_0.pt'),
                           map_location='cpu', weights_only=False)
    pivot_key = sorted(list(state_dict.keys()))[0]
    weight = state_dict[pivot_key]

    if isinstance(weight, DTensor):
        mesh_dim_names = weight.device_mesh.mesh_dim_names
        mesh = weight.device_mesh.mesh
        if 'tp' in mesh_dim_names:
            total_shards = mesh.shape[-1] * mesh.shape[-2]
            mesh_shape = (mesh.shape[-2], mesh.shape[-1])
        else:
            total_shards = mesh.shape[-1]
            mesh_shape = (mesh.shape[-1],)
    else:
        mesh_dim_names = ('fsdp',)
        total_shards = world_size
        mesh_shape = (world_size,)

    print(f'Processing {total_shards} shards with mesh_shape={mesh_shape}')

    # Load all shards
    model_state_dict_lst = [state_dict]
    model_state_dict_lst.extend([None] * (total_shards - 1))

    def load_shard(rank):
        path = os.path.join(local_dir, f'model_world_size_{world_size}_rank_{rank}.pt')
        model_state_dict_lst[rank] = torch.load(path, map_location='cpu', weights_only=False)

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
        futures = [executor.submit(load_shard, rank) for rank in range(1, total_shards)]
        for future in futures:
            future.result()

    # Merge shards
    merged = {}
    param_placements: Dict[str, List[Placement]] = {}

    for key in set(model_state_dict_lst[0].keys()):
        merged[key] = []
        for shard_dict in model_state_dict_lst:
            tensor = shard_dict.pop(key)
            if isinstance(tensor, DTensor):
                merged[key].append(tensor._local_tensor.bfloat16())
                placements = tuple(tensor.placements)
                if mesh_dim_names[0] in ('dp', 'ddp'):
                    placements = placements[1:]
                if key not in param_placements:
                    param_placements[key] = placements
            else:
                merged[key] = tensor.bfloat16()

    del model_state_dict_lst

    # Merge by placement
    for key in sorted(merged):
        if not isinstance(merged[key], list):
            continue
        if key in param_placements:
            placements = param_placements[key]
            if len(mesh_shape) == 1:
                merged[key] = merge_by_placement(merged[key], placements[0])
            else:
                raise NotImplementedError("FSDP + TP is not supported yet")
        else:
            merged[key] = torch.cat(merged[key], dim=0)

    return merged


def apply_lora_to_state_dict(base_state_dict: Dict[str, torch.Tensor],
                              lora_adapter_path: str) -> Dict[str, torch.Tensor]:
    """Apply LoRA weights to base model state dict."""
    import json
    from safetensors.torch import load_file

    print(f"Loading LoRA adapter from {lora_adapter_path}")

    # Load LoRA config
    with open(os.path.join(lora_adapter_path, "adapter_config.json"), 'r') as f:
        lora_config = json.load(f)

    lora_alpha = lora_config.get("lora_alpha", 1)
    lora_r = lora_config.get("r", 1)
    scaling = lora_alpha / lora_r

    print(f"LoRA config: r={lora_r}, alpha={lora_alpha}, scaling={scaling}")

    # Load LoRA weights
    lora_weights = load_file(os.path.join(lora_adapter_path, "adapter_model.safetensors"))

    # Group lora_A and lora_B by their target layer
    lora_pairs = {}
    for key, value in lora_weights.items():
        # Key format: base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
        if "lora_A" in key:
            base_key = key.replace(".lora_A.weight", "").replace("base_model.model.", "")
            if base_key not in lora_pairs:
                lora_pairs[base_key] = {}
            lora_pairs[base_key]["A"] = value
        elif "lora_B" in key:
            base_key = key.replace(".lora_B.weight", "").replace("base_model.model.", "")
            if base_key not in lora_pairs:
                lora_pairs[base_key] = {}
            lora_pairs[base_key]["B"] = value

    # Merge LoRA into base weights: W' = W + scaling * (B @ A)
    merged_count = 0
    for base_key, lora in lora_pairs.items():
        weight_key = base_key + ".weight"
        if weight_key in base_state_dict and "A" in lora and "B" in lora:
            A = lora["A"].to(torch.float32)  # (r, in_features)
            B = lora["B"].to(torch.float32)  # (out_features, r)
            delta = (B @ A) * scaling
            base_state_dict[weight_key] = (base_state_dict[weight_key].to(torch.float32) + delta).to(torch.bfloat16)
            merged_count += 1

    print(f"Merged {merged_count} LoRA layers into base model")
    return base_state_dict


def copy_tokenizer_files(src_dir: str, dst_dir: str):
    """Copy tokenizer files from source to destination."""
    for filename in os.listdir(src_dir):
        if filename.endswith(('.json', '.txt', '.jinja', '.model')):
            if filename.startswith(('tokenizer', 'special_tokens', 'added_tokens', 'vocab',
                                   'merges', 'chat_template', 'generation_config')):
                src = os.path.join(src_dir, filename)
                dst = os.path.join(dst_dir, filename)
                if os.path.isfile(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    print(f'Copied {filename}')


def copy_lora_adapter(src_dir: str, dst_dir: str):
    """Copy lora_adapter folder from source to destination."""
    src_lora = os.path.join(src_dir, "lora_adapter")
    dst_lora = os.path.join(dst_dir, "lora_adapter")
    if os.path.exists(src_lora):
        if os.path.exists(dst_lora):
            shutil.rmtree(dst_lora)
        shutil.copytree(src_lora, dst_lora)
        print(f'Copied lora_adapter/ folder')


def merge_checkpoint_to_hf(local_dir: str, output_dir: str, apply_lora: bool = False,
                           clean_model_prefix: bool = False) -> str:
    """Merge FSDP checkpoint to HuggingFace format.

    Args:
        local_dir: Path to checkpoint (e.g., .../global_step_84/actor)
        output_dir: Output directory for merged model
        apply_lora: If True, clean PEFT prefixes and merge LoRA into base weights (for inference).
                    If False, keep PEFT structure and copy lora_adapter/ (for resume training).
        clean_model_prefix: If True, clean PEFT prefixes but keep lora_adapter/ separate.
                            Use this for resume training with vLLM TP>1.
    """

    # Step 1: Merge FSDP shards
    state_dict = merge_fsdp_shards(local_dir)

    if apply_lora:
        # For inference/upload: clean PEFT prefixes and merge LoRA into weights
        lora_adapter_path = os.path.join(local_dir, "lora_adapter")
        if not has_lora_adapter(local_dir):
            raise ValueError(f"--lora specified but no lora_adapter folder found in {local_dir}")

        # Step 2: Clean PEFT wrapper prefixes
        print("Cleaning PEFT wrapper prefixes...")
        state_dict = clean_state_dict_keys(state_dict)

        # Step 3: Apply LoRA weights
        print("Merging LoRA weights into base model...")
        state_dict = apply_lora_to_state_dict(state_dict, lora_adapter_path)
    elif clean_model_prefix:
        # For resume training with vLLM TP>1: clean PEFT prefixes but keep lora_adapter/ separate
        print("Cleaning PEFT wrapper prefixes (keeping lora_adapter/ separate)...")
        state_dict = clean_state_dict_keys(state_dict)
    else:
        # For resume training: keep PEFT structure as-is
        print("Keeping PEFT structure for resume training...")

    # Step 4: Save as HuggingFace model
    print('Writing to local disk')
    os.makedirs(output_dir, exist_ok=True)
    config = AutoConfig.from_pretrained(local_dir)

    if 'ForTokenClassification' in config.architectures[0]:
        auto_model = AutoModelForTokenClassification
    elif 'ForCausalLM' in config.architectures[0]:
        auto_model = AutoModelForCausalLM
    elif 'ForConditionalGeneration' in config.architectures[0]:
        auto_model = AutoModelForVision2Seq
    else:
        raise NotImplementedError(f'Unknown architecture {config.architectures}')

    with torch.device('meta'):
        model = auto_model.from_config(config, torch_dtype=torch.bfloat16)
    model.to_empty(device='cpu')

    print(f'Saving model to {output_dir}')
    model.save_pretrained(output_dir, state_dict=state_dict)
    del state_dict
    del model

    # Step 5: Copy tokenizer files
    copy_tokenizer_files(local_dir, output_dir)

    # Step 6: Copy lora_adapter/ if not merging LoRA into weights (for resume training)
    if (not apply_lora) and has_lora_adapter(local_dir):
        copy_lora_adapter(local_dir, output_dir)

    return output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge FSDP/LoRA checkpoints to HuggingFace format")
    parser.add_argument('--local_dir', required=True, type=str,
                       help="Path to checkpoint (e.g., .../global_step_84/actor)")
    parser.add_argument("--hf_upload_path", type=str,
                       help="HuggingFace repo (e.g., 'username/model-name')")
    parser.add_argument("--private", action="store_true", help="Upload as private repo")
    parser.add_argument("--output_dir", type=str, help="Output directory (default: local_dir/huggingface)")
    parser.add_argument("--lora", action="store_true",
                       help="Merge LoRA weights into base model (for inference/upload). "
                            "Without this flag, PEFT structure is preserved for resume training.")
    parser.add_argument("--clean-model-prefix", action="store_true",
                       help="Clean PEFT prefixes (base_model.model.) but keep lora_adapter/ separate. "
                            "Use this for resume training with vLLM tensor_parallel_size > 1.")
    args = parser.parse_args()

    if args.lora and args.clean_model_prefix:
        parser.error("--lora and --clean-model-prefix are mutually exclusive")

    local_dir = args.local_dir
    output_dir = args.output_dir or os.path.join(local_dir, 'huggingface')

    hf_path = merge_checkpoint_to_hf(local_dir, output_dir, apply_lora=args.lora,
                                     clean_model_prefix=args.clean_model_prefix)

    if args.hf_upload_path:
        print(f"Uploading to HuggingFace: {args.hf_upload_path}")
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=args.hf_upload_path, private=args.private, exist_ok=True)
        api.upload_folder(folder_path=hf_path, repo_id=args.hf_upload_path, repo_type="model")
        print(f"Done: https://huggingface.co/{args.hf_upload_path}")
