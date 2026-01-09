#!/usr/bin/env python3
"""
Compute synflow scores for all 800 cell architecture configurations.

This script:
1. Reads mamba_configs.txt with 800 architecture configurations
2. Creates MambaEncoder model for each configuration
3. Computes synflow score for each model using actual test_clean data shape
4. Runs 5 times with different batch data and averages the scores
5. Saves results to output file
"""

import sys
import ast
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import traceback
import random

# Add espnet to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from espnet2.asr.encoder.mamba_encoder import MambaEncoder


def get_layer_metric_array(model: nn.Module, metric_fn, mode='param'):
    """
    Helper function to collect metric values from all layers.
    Following test_zero_cost_proxy.py implementation.
    
    Args:
        model: PyTorch model
        metric_fn: Function that takes a layer and returns a metric tensor
        mode: 'param' or 'channel' (for now, only 'param' is used)
    
    Returns:
        List of metric values (one per layer with weight)
    """
    metric_array = []
    for module in model.modules():
        # Check if module has weight attribute (e.g., Linear, Conv, etc.)
        if hasattr(module, 'weight') and module.weight is not None:
            metric_value = metric_fn(module)
            if metric_value is not None:
                metric_array.append(metric_value)
    return metric_array


def compute_synflow(model: nn.Module, inputs: torch.Tensor, ilens: torch.Tensor) -> float:
    """
    Compute Synflow score (zero-cost proxy) following test_zero_cost_proxy.py implementation.
    
    Synflow mechanism:
    1. Convert all params to their abs (keep sign for converting back)
    2. Use input of ones
    3. Compute gradients with backward pass
    4. Calculate |weight * weight.grad| for each layer
    5. Restore original parameter signs
    6. Scale by number of parameters for fair comparison across different model sizes
    
    Args:
        model: PyTorch model (MambaEncoder)
        inputs: Input tensor (batch, seq_len, input_dim) - used to get shape
        ilens: Input lengths (batch,) - used for MambaEncoder forward
    
    Returns:
        Synflow score (sum of all layer metrics, normalized by number of parameters)
    """
    device = inputs.device
    
    # Set model to eval mode
    model.eval()
    
    # Note: Cannot use double() because Mamba CUDA kernels only support float32/float16/bfloat16
    # Keep original dtype (typically float32) for compatibility with CUDA kernels
    
    # Convert params to their abs. Keep sign for converting it back.
    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.named_parameters():
            signs[name] = torch.sign(param.data)
            param.data.abs_()
        return signs
    
    # Convert to orig values
    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.named_parameters():
            if 'weight_mask' not in name:
                param.data.mul_(signs[name])
    
    # Keep signs of all params
    signs = linearize(model)
    
    try:
        # Compute gradients with input of 1s
        model.zero_grad()
        
        # Create input of ones with same shape as inputs (but batch_size=1)
        # MambaEncoder expects (batch, seq_len, input_dim) and ilens
        # Use first sample's shape as reference
        seq_len = inputs.shape[1]  # Get seq_len from inputs
        input_dim = inputs.shape[2]  # Get input_dim from inputs
        # Use same dtype as model parameters (typically float32, as double is not supported by Mamba CUDA kernels)
        param_dtype = next(model.parameters()).dtype
        inputs_ones = torch.ones([1, seq_len, input_dim], dtype=param_dtype, device=device)
        ilens_ones = torch.tensor([seq_len], dtype=torch.long, device=device)
        
        # Forward pass
        # MambaEncoder.forward returns (output, olens, prev_states)
        output, _, _ = model.forward(inputs_ones, ilens_ones)
        
        # Backward pass
        torch.sum(output).backward()
        
        # Select the gradients that we want to use for search/prune
        def synflow(layer):
            if hasattr(layer, 'weight') and layer.weight is not None:
                if layer.weight.grad is not None:
                    return torch.abs(layer.weight * layer.weight.grad)
                else:
                    return torch.zeros_like(layer.weight)
            return None
        
        # Get metric array for all layers
        grads_abs = get_layer_metric_array(model, synflow, mode='param')
        
        # Sum all layer metrics to get total synflow score
        if len(grads_abs) > 0:
            synflow_score = sum([grad.sum().item() for grad in grads_abs])
        else:
            synflow_score = 0.0
        
        # Scale by number of parameters for fair comparison across models with different sizes
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_weight_params = sum(p.numel() for n, p in model.named_parameters() if 'weight' in n)
        if num_weight_params > 0:
            synflow_score = synflow_score / num_weight_params
        
        # Apply signs of all params
        nonlinearize(model, signs)
        
    except Exception as e:
        print(f"      Error in synflow computation: {e}")
        import traceback
        traceback.print_exc()
        synflow_score = float('nan')
        # Restore signs even if error occurred
        try:
            nonlinearize(model, signs)
        except:
            pass
    
    return synflow_score


def parse_config_file(config_file: str) -> List[Tuple[List[str], float, int]]:
    """
    Parse mamba_configs.txt file.
    
    Format: ['I', 'I', 'I', 'C', 'M']\t0.688\t29
    
    Returns:
        List of (module_config, score, count) tuples
    """
    configs = []
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Split by tab
            parts = line.split('\t')
            if len(parts) < 1:
                continue
            
            # Parse module config (first part)
            try:
                module_config = ast.literal_eval(parts[0])
                score = float(parts[1]) if len(parts) > 1 else None
                count = int(parts[2]) if len(parts) > 2 else None
                configs.append((module_config, score, count))
            except Exception as e:
                print(f"Warning: Failed to parse line: {line}, error: {e}")
                continue
    
    return configs


def create_mamba_encoder(module_config: List[str], num_blocks: int = 12, device: str = "cuda") -> MambaEncoder:
    """
    Create MambaEncoder with given module_config.
    
    Args:
        module_config: List of 5 module initials (e.g., ['I', 'I', 'I', 'C', 'M'])
        num_blocks: Number of encoder blocks
        device: Device to create model on
    
    Returns:
        MambaEncoder model
    """
    # Expected number of modules is 5
    if len(module_config) != 5:
        raise ValueError(f"module_config must have exactly 5 modules, got {len(module_config)}")
    
    # Use the 5-module config directly (MambaEncoder/MambaCellBlock expects 5 modules)
    # Repeat the same config for all blocks
    module_configs = [module_config] * num_blocks
    
    # Create encoder with settings from train_asr.yaml
    encoder = MambaEncoder(
        input_size=80,  # Standard mel spectrogram dimension
        output_size=256,
        num_blocks=num_blocks,
        dropout_rate=0.1,
        positional_dropout_rate=0.05,
        input_layer="conv2d",
        normalize_before=True,
        rel_pos_type="latest",
        pos_enc_layer_type="rel_pos",
        activation_type="swish",
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        linear_units=1024,
        cnn_module_kernel=31,
        module_configs=module_configs,
    ).to(device)
    
    return encoder


def main():
    """Main function to compute synflow scores for all configurations."""
    
    # Paths
    script_dir = Path(__file__).parent
    config_file = script_dir.parent.parent.parent.parent / "mamba_configs.txt"
    output_file = script_dir / "synflow_scores_avg.txt"
    
    print("=" * 80)
    print("Computing Synflow Scores for Mamba Architectures")
    print("=" * 80)
    print(f"Config file: {config_file}")
    print(f"Output file: {output_file}")
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Parse config file
    print(f"\n1. Parsing config file...")
    configs = parse_config_file(config_file)
    print(f"   Found {len(configs)} configurations")
    
    # Configure simulated data for synflow (input of ones)
    print(f"\n2. Preparing simulated input data (ones)...")
    batch_size = 8
    input_dim = 80  # Matches mel feature dimension / encoder input_size
    min_seq_len = 800
    max_seq_len = 1200
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length range: [{min_seq_len}, {max_seq_len}] frames")
    print(f"   Feature dimension: {input_dim}")
    
    # Storage for scores across 5 runs (each with different batch data)
    # Key: config_str, Value: list of (synflow_score, num_params, status) tuples
    scores_dict = {}
    num_params_dict = {}  # Store num_params for each config (should be same across runs)
    
    # Run 5 times with different batch data
    num_runs = 5
    print(f"\n3. Computing synflow scores (5 runs with different batch data)...")
    print(f"   Processing {len(configs)} configurations across {num_runs} runs...")
    
    for run_idx in range(num_runs):
        print(f"\n   {'='*70}")
        print(f"   Run {run_idx+1}/{num_runs}")
        print(f"   {'='*70}")
        
        # Generate simulated ones input with random sequence length within range
        seq_len = random.randint(min_seq_len, max_seq_len)
        x_test = torch.ones(
            batch_size,
            seq_len,
            input_dim,
            dtype=torch.float32,
            device=device,
        )
        ilens = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
        
        print(f"   Using simulated ones input: seq_len={seq_len}, batch_size={batch_size}, input_dim={input_dim}")
        
        # Process each configuration
        for idx, (module_config, old_score, count) in enumerate(configs):
            config_str = str(module_config)
            
            # Initialize dict entry if first run
            if config_str not in scores_dict:
                scores_dict[config_str] = []
            
            # Use count as num_blocks (count represents number of blocks for this architecture)
            if count is None:
                num_blocks = 12
            else:
                num_blocks = count
            
            try:
                # Create model with count as num_blocks
                model = create_mamba_encoder(module_config, num_blocks=num_blocks, device=device)
                
                # Count parameters (only store once, should be same across runs)
                if config_str not in num_params_dict:
                    num_params = sum(p.numel() for p in model.parameters())
                    num_params_dict[config_str] = num_params
                else:
                    num_params = num_params_dict[config_str]
                
                # Test forward pass first
                with torch.no_grad():
                    try:
                        # MambaEncoder.forward returns (output, olens, prev_states)
                        output, olens, _ = model(x_test, ilens)
                    except Exception as e:
                        print(f"      [{idx+1}/{len(configs)}] {config_str}: Forward pass failed: {e}")
                        scores_dict[config_str].append((np.nan, "forward_failed"))
                        del model
                        torch.cuda.empty_cache() if device.type == "cuda" else None
                        continue
                
                # Compute synflow
                synflow_score = compute_synflow(model, x_test, ilens)
                
                if not np.isnan(synflow_score):
                    scores_dict[config_str].append((synflow_score, "success"))
                    print(f"      [{idx+1}/{len(configs)}] {config_str}: Synflow={synflow_score:.6f}")
                else:
                    scores_dict[config_str].append((np.nan, "nan"))
                    print(f"      [{idx+1}/{len(configs)}] {config_str}: Synflow=NaN")
                
                # Clean up
                del model
                torch.cuda.empty_cache() if device.type == "cuda" else None
                
            except Exception as e:
                print(f"      [{idx+1}/{len(configs)}] {config_str}: Error: {e}")
                scores_dict[config_str].append((np.nan, f"error: {str(e)[:50]}"))
                torch.cuda.empty_cache() if device.type == "cuda" else None
    
    # Calculate averages for each configuration
    print(f"\n4. Calculating average synflow scores across {num_runs} runs...")
    results = []
    
    for idx, (module_config, old_score, count) in enumerate(configs):
        config_str = str(module_config)
        
        # Get all scores for this config (exclude NaN)
        synflow_scores = [score for score, status in scores_dict[config_str] if not np.isnan(score)]
        
        # Calculate average (exclude NaN values)
        if len(synflow_scores) > 0:
            synflow_avg = np.mean(synflow_scores)
            status = "success"
        else:
            # All runs resulted in NaN
            synflow_avg = np.nan
            status = "all_nan"
        
        num_params = num_params_dict.get(config_str, 0)
        results.append((config_str, synflow_avg, old_score, count, num_params, status))
        
        if (idx + 1) % 100 == 0:  # Print progress every 100 configs
            if not np.isnan(synflow_avg):
                print(f"   [{idx+1}/{len(configs)}] {config_str}: avg={synflow_avg:.6f} (from {len(synflow_scores)}/{num_runs} runs)")
            else:
                print(f"   [{idx+1}/{len(configs)}] {config_str}: NaN (all {num_runs} runs failed)")
    
    # Save results
    print(f"\n5. Saving results to {output_file}...")
    with open(output_file, 'w') as f:
        # Header
        f.write("# Format: module_config\tsynflow_score\tnum_params\tN of Blocks\ttotal_params\tstatus\n")
        
        # Sort by synflow score (descending, NaN last)
        sorted_results = sorted(
            results,
            key=lambda x: (np.isnan(x[1]), -x[1] if not np.isnan(x[1]) else 0)
        )
        
        # Write results
        for config_str, synflow, old_score, count, num_params, status in sorted_results:
            if np.isnan(synflow):
                synflow_str = "NaN"
            else:
                synflow_str = f"{synflow:.6f}"
            
            old_score_str = f"{old_score:.3f}" if old_score is not None else "None"
            count_str = str(count) if count is not None else "None"
            
            f.write(f"{config_str}\t{synflow_str}\t{old_score_str}\t{count_str}\t{num_params}\t{status}\n")
    
    # Print summary
    print(f"\n6. Summary:")
    successful = sum(1 for r in results if not np.isnan(r[1]))
    failed = len(results) - successful
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    
    if successful > 0:
        valid_scores = [r[1] for r in results if not np.isnan(r[1])]
        print(f"   Min synflow: {min(valid_scores):.6f}")
        print(f"   Max synflow: {max(valid_scores):.6f}")
        print(f"   Mean synflow: {np.mean(valid_scores):.6f}")
    
    print(f"\nResults saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

