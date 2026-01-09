#!/usr/bin/env python3
"""
Compute jacob_cov scores for all 800 cell architecture configurations.

This script:
1. Reads mamba_configs.txt with 800 architecture configurations
2. Creates MambaEncoder model for each configuration
3. Computes jacob_cov score for each model using actual test_clean data
4. Saves results to output file
"""

import sys
import os
import ast
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import traceback
import random
import kaldiio
import soundfile

# Add espnet to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from espnet2.asr.encoder.mamba_encoder import MambaEncoder
from espnet2.asr.frontend.default import DefaultFrontend


def get_batch_jacobian(model: nn.Module, x: torch.Tensor, ilens: torch.Tensor, split_data: int = 1):
    """
    Compute Jacobian matrix for a batch of inputs.
    Following test_zero_cost_proxy.py implementation.
    
    Args:
        model: PyTorch model (MambaEncoder)
        x: Input tensor (batch, seq_len, input_dim)
        ilens: Input lengths (batch,)
        split_data: Number of splits for processing large batches
    
    Returns:
        jacob: Jacobian matrix (batch, ...) - gradient of output w.r.t. input
    """
    # Create a copy with requires_grad=True to ensure gradient computation
    x = x.clone().detach().requires_grad_(True)
    
    N = x.shape[0]
    
    # Process each split and accumulate gradients
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        
        # Forward pass on split
        # MambaEncoder.forward returns (output, olens, prev_states)
        y, _, _ = model(x[st:en], ilens[st:en])
        
        # Backward pass: compute gradient of each output element w.r.t. input
        # torch.ones_like(y) creates a tensor of ones with same shape as y
        # This computes the gradient for each output element independently
        y.backward(torch.ones_like(y), retain_graph=(sp < split_data - 1))
    
    jacob = x.grad.detach()
    x.requires_grad_(False)
    return jacob


def eval_score(jacob: np.ndarray, labels=None):
    """
    Evaluate jacob_cov score from Jacobian matrix.
    Following test_zero_cost_proxy.py implementation exactly.
    
    Args:
        jacob: Jacobian matrix (batch, flattened_features)
        labels: Not used in this implementation
    
    Returns:
        (jacob_cov score, nan_reason) tuple
        nan_reason: None if valid, or one of:
            - "zero_variance": All features have zero variance
            - "insufficient_samples": Less than 2 samples for correlation computation
            - "corr_nan": Correlation matrix contains NaN
            - "corr_inf": Correlation matrix contains Inf
    """
    # Remove features with zero variance (std == 0) to avoid division by zero in corrcoef
    std = np.std(jacob, axis=0)
    valid_features = std > 1e-8  # Features with non-zero variance
    num_valid = np.sum(valid_features)
    num_total = jacob.shape[1]
    
    if num_valid == 0:
        # All features have zero variance, return NaN
        return np.nan, "zero_variance"
    
    if num_valid < num_total:
        # Filtered features (debug info removed for production)
        pass
    
    jacob_filtered = jacob[:, valid_features]
    
    # Check if we have enough samples for correlation computation
    # Need at least 2 samples for correlation
    if jacob_filtered.shape[0] < 2:
        return np.nan, "insufficient_samples"
    
    # Compute correlation matrix (suppress warnings as we handle NaN/inf below)
    with np.errstate(divide='ignore', invalid='ignore'):
        corrs = np.corrcoef(jacob_filtered)
    
    # Handle NaN/inf in correlation matrix (shouldn't happen after filtering, but just in case)
    if np.any(np.isnan(corrs)):
        return np.nan, "corr_nan"
    
    if np.any(np.isinf(corrs)):
        return np.nan, "corr_inf"
    
    # Symmetrize correlation matrix for numerical stability
    corrs = (corrs + corrs.T) / 2.0
    
    # Compute eigenvalues (using eigvalsh for symmetric matrices - faster and more stable)
    v = np.linalg.eigvalsh(corrs)  # returns sorted real eigenvalues
    
    # Clip negative eigenvalues to zero (correlation matrix should be positive semi-definite)
    v = np.clip(v, 0.0, None)
    
    # Compute score: -sum(log(v + k) + 1/(v + k))
    k = 1e-5
    score = -np.sum(np.log(v + k) + 1. / (v + k))
    return score, None


def compute_jacob_cov(model: nn.Module, inputs: torch.Tensor, ilens: torch.Tensor, split_data: int = 1) -> Tuple[float, Optional[str]]:
    """
    Compute Jacobian Covariance score (zero-cost proxy).
    Following test_zero_cost_proxy.py implementation exactly.
    
    Args:
        model: PyTorch model (MambaEncoder)
        inputs: Input tensor (batch, seq_len, input_dim)
        ilens: Input lengths (batch,)
        split_data: Number of splits for processing large batches
    
    Returns:
        (jacob_cov score, nan_reason) tuple
        nan_reason: None if valid, or one of:
            - "zero_variance": All features have zero variance
            - "insufficient_samples": Less than 2 samples for correlation computation
            - "corr_nan": Correlation matrix contains NaN
            - "corr_inf": Correlation matrix contains Inf
            - "jacobian_error": Error during jacobian computation
    """
    device = inputs.device
    
    # Set model to eval mode
    model.eval()
    
    # Compute gradients (but don't apply them)
    model.zero_grad()
    
    try:
        # Get batch Jacobian
        jacobs = get_batch_jacobian(model, inputs, ilens, split_data=split_data)
        
        # Reshape to (batch_size, flattened_features)
        jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
        
        # Evaluate score
        jc, nan_reason = eval_score(jacobs)
        
    except Exception as e:
        print(f"      Error in jacob_cov computation: {e}")
        import traceback
        traceback.print_exc()
        jc = np.nan
        nan_reason = "jacobian_error"
    
    return jc, nan_reason


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
    """Main function to compute jacob_cov scores for all configurations."""
    
    # Paths
    script_dir = Path(__file__).parent
    config_file = script_dir.parent.parent.parent.parent / "mamba_configs.txt"
    output_file = script_dir / "jacob_cov_scores_avg.txt"
    
    print("=" * 80)
    print("Computing Jacob Covariance Scores for Mamba Architectures")
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
    
    # Load test_clean audio data
    print(f"\n2. Loading test_clean audio data...")
    batch_size = 16
    dump_dir = script_dir / "dump" / "raw" / "test_clean"
    wav_scp = dump_dir / "wav.scp"
    
    if not wav_scp.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_scp}. Please run data preparation first.")
    
    # Create frontend for feature extraction (mel spectrogram)
    # Using same settings as train_asr.yaml
    frontend = DefaultFrontend(
        n_fft=512,
        hop_length=160,
        win_length=None,  # Will use n_fft
        fs=16000,  # 16kHz sampling rate
    ).to(device)
    frontend.eval()
    
    # Load audio data from wav.scp
    audio_data = []
    with open(wav_scp, 'r') as f:
        for line in f:
            utt_id, wav_path = line.strip().split(None, 1)
            # Load audio using kaldiio (supports ark format)
            audio = kaldiio.load_mat(wav_path)
            # kaldiio returns (rate, array) for sound files
            if isinstance(audio, tuple) and len(audio) == 2:
                rate, waveform = audio
            else:
                # If it's just array, assume 16kHz
                waveform = audio
                rate = 16000
            
            audio_data.append((utt_id, waveform, rate))
    
    print(f"   Loaded {len(audio_data)} utterances from test_clean")
    
    # Storage for scores across 5 runs (each with different batch data)
    # Key: config_str, Value: list of (jacob_cov_score, num_params, status) tuples
    scores_dict = {}
    num_params_dict = {}  # Store num_params for each config (should be same across runs)
    
    # Run 5 times with different batch data
    num_runs = 5
    print(f"\n3. Computing jacob_cov scores (5 runs with different batch data)...")
    print(f"   Processing {len(configs)} configurations across {num_runs} runs...")
    
    for run_idx in range(num_runs):
        print(f"\n   {'='*70}")
        print(f"   Run {run_idx+1}/{num_runs}")
        print(f"   {'='*70}")
        
        # Sample 8 sequences randomly (different for each run)
        sampled_data = random.sample(audio_data, min(batch_size, len(audio_data)))
        
        # Extract features using frontend
        print(f"   Sampling and extracting features from {len(sampled_data)} utterances...")
        feats_data = []
        with torch.no_grad():
            for utt_id, waveform, rate in sampled_data:
                # Convert to torch tensor: (seq_len,) -> (1, seq_len)
                waveform_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(device)
                # Create input_lengths tensor: (1,)
                input_lengths = torch.tensor([waveform.shape[0]], dtype=torch.long).to(device)
                
                # Extract features: (1, seq_len) -> (1, feat_len, feat_dim)
                feat, feat_lengths = frontend(waveform_tensor, input_lengths)
                # Remove batch dimension: (1, feat_len, feat_dim) -> (feat_len, feat_dim)
                feat = feat.squeeze(0).cpu().numpy()
                
                feats_data.append((utt_id, feat))
        
        # Prepare batch
        max_len = max(feat.shape[0] for _, feat in feats_data)
        input_dim = feats_data[0][1].shape[1]
        
        print(f"   Batch size: {batch_size}, Max sequence length: {max_len}, Input dimension: {input_dim}")
        
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
                
                # Prepare batch from feats_data (already extracted features)
                batch_feats = []
                batch_ilens = []
                for utt_id, feat in feats_data:
                    # Pad to max_len
                    padded_feat = np.pad(feat, ((0, max_len - feat.shape[0]), (0, 0)), mode='constant', constant_values=0)
                    batch_feats.append(padded_feat)
                    batch_ilens.append(feat.shape[0])
                
                x_test = torch.tensor(np.array(batch_feats), dtype=torch.float32, device=device)
                ilens = torch.tensor(batch_ilens, dtype=torch.long, device=device)
                
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
                
                # Compute jacob_cov
                jacob_cov_score, _ = compute_jacob_cov(model, x_test, ilens, split_data=1)
                
                if not np.isnan(jacob_cov_score):
                    scores_dict[config_str].append((jacob_cov_score, "success"))
                    if (idx + 1) % 50 == 0:  # Print progress every 50 configs
                        print(f"      [{idx+1}/{len(configs)}] {config_str}: {jacob_cov_score:.6f}")
                else:
                    scores_dict[config_str].append((np.nan, "nan"))
                
                # Clean up
                del model
                torch.cuda.empty_cache() if device.type == "cuda" else None
                
            except Exception as e:
                print(f"      [{idx+1}/{len(configs)}] {config_str}: Error: {e}")
                scores_dict[config_str].append((np.nan, f"error: {str(e)[:50]}"))
                torch.cuda.empty_cache() if device.type == "cuda" else None
    
    # Calculate averages for each configuration
    print(f"\n4. Calculating average jacob_cov scores across {num_runs} runs...")
    results = []
    
    for idx, (module_config, old_score, count) in enumerate(configs):
        config_str = str(module_config)
        
        # Get all scores for this config (exclude NaN)
        jacob_cov_scores = [score for score, status in scores_dict[config_str] if not np.isnan(score)]
        
        # Calculate average (exclude NaN values)
        if len(jacob_cov_scores) > 0:
            jacob_cov_avg = np.mean(jacob_cov_scores)
            status = "success"
        else:
            # All runs resulted in NaN
            jacob_cov_avg = np.nan
            status = "all_nan"
        
        num_params = num_params_dict.get(config_str, 0)
        results.append((config_str, jacob_cov_avg, old_score, count, num_params, status))
        
        if (idx + 1) % 100 == 0:  # Print progress every 100 configs
            if not np.isnan(jacob_cov_avg):
                print(f"   [{idx+1}/{len(configs)}] {config_str}: avg={jacob_cov_avg:.6f} (from {len(jacob_cov_scores)}/{num_runs} runs)")
            else:
                print(f"   [{idx+1}/{len(configs)}] {config_str}: NaN (all {num_runs} runs failed)")
    
    # Save results
    print(f"\n4. Saving results to {output_file}...")
    with open(output_file, 'w') as f:
        # Header
        f.write("# Format: module_config\tjacob_cov_score\tnum_params\tN of Blocks\ttotal_params\tstatus\n")
        
        # Sort by jacob_cov score (descending, NaN last)
        sorted_results = sorted(
            results,
            key=lambda x: (np.isnan(x[1]), -x[1] if not np.isnan(x[1]) else 0)
        )
        
        # Write results
        for config_str, jacob_cov, old_score, count, num_params, status in sorted_results:
            if np.isnan(jacob_cov):
                jacob_cov_str = "NaN"
            else:
                jacob_cov_str = f"{jacob_cov:.6f}"
            
            old_score_str = f"{old_score:.3f}" if old_score is not None else "None"
            count_str = str(count) if count is not None else "None"
            
            f.write(f"{config_str}\t{jacob_cov_str}\t{old_score_str}\t{count_str}\t{num_params}\t{status}\n")
    
    # Print summary
    print(f"\n5. Summary:")
    successful = sum(1 for r in results if not np.isnan(r[1]))
    failed = len(results) - successful
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    
    if successful > 0:
        valid_scores = [r[1] for r in results if not np.isnan(r[1])]
        print(f"   Min jacob_cov: {min(valid_scores):.6f}")
        print(f"   Max jacob_cov: {max(valid_scores):.6f}")
        print(f"   Mean jacob_cov: {np.mean(valid_scores):.6f}")
    
    print(f"\nResults saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

