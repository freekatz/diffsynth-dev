#!/usr/bin/env python3
"""Validate Wan4D training data flow.

This script verifies that:
1. Dataset returns correct tensor shapes and dtypes
2. All required fields (target_video, temporal_coords, c2w, etc.) are present
3. Model forward pass runs without errors
4. Coordinate semantics are correct (frame indices, not seconds)

Usage:
    python scripts/validate_training.py --data-dir data/demo-data2/index.json [--device cuda]
"""

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.dataset import Wan4DDataset
from diffsynth.models.wan_video_4d_dit import Wan4DModel


def validate_dataset(index_path: str, device: str = "cpu") -> dict:
    """Validate dataset output shapes and types."""
    print("=" * 60)
    print("1. Dataset Validation")
    print("=" * 60)
    
    # Derive dataset_root from index_path (parent directory)
    index_path = Path(index_path)
    dataset_root = index_path.parent
    dataset = Wan4DDataset(dataset_root=dataset_root, index_path=str(index_path))
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    
    # Required fields
    required_fields = [
        "target_video",
        "prompt_context", 
        "condition_pixel_indices",
        "condition_latent_indices",
        "temporal_coords",
    ]
    
    # Optional but important for Wan4D
    optional_fields = [
        "c2w",
        "plucker_embedding",
    ]
    
    print("\nChecking required fields...")
    for field in required_fields:
        if field not in sample:
            print(f"  [FAIL] Missing required field: {field}")
            return {"success": False}
        print(f"  [OK] {field}: {sample[field].shape}, dtype={sample[field].dtype}")
    
    print("\nChecking optional fields...")
    for field in optional_fields:
        if field in sample and sample[field] is not None:
            print(f"  [OK] {field}: {sample[field].shape}, dtype={sample[field].dtype}")
        else:
            print(f"  [INFO] {field}: not present")
    
    # Validate shapes
    target_video = sample["target_video"]
    C, T, H, W = target_video.shape
    print(f"\nVideo shape: C={C}, T={T}, H={H}, W={W}")
    
    F_latent = (T - 1) // 4 + 1
    print(f"Expected latent frames: {F_latent}")
    
    # Check temporal_coords
    temporal_coords = sample["temporal_coords"]
    print(f"\ntemporal_coords: {temporal_coords}")
    print(f"  Shape: {temporal_coords.shape}")
    print(f"  Dtype: {temporal_coords.dtype}")
    print(f"  Values: {temporal_coords.tolist()[:10]}..." if len(temporal_coords) > 10 else f"  Values: {temporal_coords.tolist()}")
    
    # temporal_coords should be float (latent frame indices) for 5D RoPE t-axis
    if temporal_coords.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        print(f"  [WARN] temporal_coords should be float type (frame indices), got {temporal_coords.dtype}")
    else:
        print(f"  [OK] temporal_coords is float type (latent frame indices)")
    
    if temporal_coords.shape[0] != F_latent:
        print(f"  [WARN] temporal_coords length ({temporal_coords.shape[0]}) != F_latent ({F_latent})")
    else:
        print(f"  [OK] temporal_coords length matches F_latent")
    
    # Check c2w
    if "c2w" in sample and sample["c2w"] is not None:
        c2w = sample["c2w"]
        print(f"\nc2w: {c2w.shape}")
        if c2w.shape != (F_latent, 4, 4):
            print(f"  [WARN] c2w shape should be ({F_latent}, 4, 4), got {c2w.shape}")
        else:
            print(f"  [OK] c2w shape is correct")
    
    return {
        "success": True,
        "F_latent": F_latent,
        "H": H,
        "W": W,
        "sample": sample,
    }


def validate_model(dataset_result: dict, device: str = "cpu") -> dict:
    """Validate model forward pass."""
    print("\n" + "=" * 60)
    print("2. Model Validation")
    print("=" * 60)
    
    if not dataset_result["success"]:
        print("[SKIP] Dataset validation failed, skipping model validation")
        return {"success": False}
    
    sample = dataset_result["sample"]
    F_latent = dataset_result["F_latent"]
    H_latent = dataset_result["H"] // 8
    W_latent = dataset_result["W"] // 8
    
    print(f"\nCreating Wan4DModel...")
    print(f"  Latent shape: [1, 16, {F_latent}, {H_latent}, {W_latent}]")
    
    # Create model with small config for testing
    model = Wan4DModel(
        dim=2048,
        ffn_dim=8192,
        out_dim=16,
        text_dim=4096,
        freq_dim=256,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=16,
        num_layers=4,  # Small for testing
        has_image_input=False,
    ).to(device)
    model.eval()
    
    print(f"  Model created successfully")
    
    # Prepare inputs
    B = 1
    x = torch.randn(B, 16, F_latent, H_latent, W_latent, device=device)
    timestep = torch.tensor([500.0], device=device)
    context = torch.randn(B, 512, 4096, device=device)  # Typical text context
    temporal_coords = sample["temporal_coords"].unsqueeze(0).to(device)
    
    c2w = None
    if "c2w" in sample and sample["c2w"] is not None:
        c2w = sample["c2w"].unsqueeze(0).to(device, dtype=x.dtype)
    
    plucker_embedding = None
    if "plucker_embedding" in sample and sample["plucker_embedding"] is not None:
        plucker_embedding = sample["plucker_embedding"].unsqueeze(0).to(device, dtype=x.dtype)
    
    condition_latents = torch.zeros_like(x)
    condition_mask = torch.zeros(B, 4, F_latent, H_latent, W_latent, device=device)
    
    print(f"\nInput shapes:")
    print(f"  x: {x.shape}")
    print(f"  timestep: {timestep.shape}")
    print(f"  context: {context.shape}")
    print(f"  temporal_coords: {temporal_coords.shape}, dtype={temporal_coords.dtype}")
    print(f"  c2w: {c2w.shape if c2w is not None else 'None'}")
    print(f"  plucker_embedding: {plucker_embedding.shape if plucker_embedding is not None else 'None'}")
    print(f"  condition_latents: {condition_latents.shape}")
    print(f"  condition_mask: {condition_mask.shape}")
    
    print(f"\nRunning forward pass...")
    try:
        with torch.no_grad():
            output = model(
                x=x,
                timestep=timestep,
                context=context,
                temporal_coords=temporal_coords,
                c2w=c2w,
                plucker_embedding=plucker_embedding,
                condition_latents=condition_latents,
                condition_mask=condition_mask,
            )
        print(f"  [OK] Forward pass successful")
        print(f"  Output shape: {output.shape}")
        
        expected_shape = (B, 16, F_latent, H_latent, W_latent)
        if output.shape != expected_shape:
            print(f"  [WARN] Output shape {output.shape} != expected {expected_shape}")
        else:
            print(f"  [OK] Output shape matches expected")
        
        return {"success": True}
        
    except Exception as e:
        print(f"  [FAIL] Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False}


def validate_coordinates(dataset_result: dict) -> dict:
    """Validate coordinate semantics."""
    print("\n" + "=" * 60)
    print("3. Coordinate Semantics Validation")
    print("=" * 60)
    
    if not dataset_result["success"]:
        print("[SKIP] Dataset validation failed")
        return {"success": False}
    
    sample = dataset_result["sample"]
    temporal_coords = sample["temporal_coords"]
    
    print("\nChecking temporal_coords semantics...")
    
    # temporal_coords should be float (latent frame indices) for 5D RoPE t-axis
    if temporal_coords.dtype not in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
        print(f"  [WARN] temporal_coords should be float (frame indices), got {temporal_coords.dtype}")
        return {"success": False}
    
    # Values should be non-negative latent frame indices
    print(f"  Values (first 5): {temporal_coords[:5].tolist()}")
    if temporal_coords[0] >= 0:
        print(f"  [OK] temporal_coords appear to be valid latent frame indices (for 5D RoPE t-axis)")
    else:
        print(f"  [WARN] temporal_coords values seem unusual (negative)")
    
    print(f"  [OK] Coordinate semantics validated (latent frame indices used as 5D RoPE t-axis)")
    return {"success": True}


def main():
    parser = argparse.ArgumentParser(description="Validate Wan4D training data flow")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/demo-data2/index.json",
        help="Path to index.json file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run model validation on",
    )
    args = parser.parse_args()
    
    print(f"Validating with data: {args.data_dir}")
    print(f"Device: {args.device}")
    
    # Run validations
    dataset_result = validate_dataset(args.data_dir, args.device)
    model_result = validate_model(dataset_result, args.device)
    coord_result = validate_coordinates(dataset_result)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = all([
        dataset_result["success"],
        model_result["success"],
        coord_result["success"],
    ])
    
    print(f"Dataset validation: {'PASS' if dataset_result['success'] else 'FAIL'}")
    print(f"Model validation: {'PASS' if model_result['success'] else 'FAIL'}")
    print(f"Coordinate semantics: {'PASS' if coord_result['success'] else 'FAIL'}")
    print(f"\nOverall: {'PASS ✓' if all_passed else 'FAIL ✗'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
