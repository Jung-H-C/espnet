#!/usr/bin/env python3
"""Test script to check if CUDA kernels are properly installed."""

import sys
import torch

# Check Python version
python_version = sys.version_info
print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
if python_version < (3, 10):
    print("⚠️  Warning: Python < 3.10 may have compatibility issues with some packages")

print("=" * 60)
print("CUDA Kernel Installation Test")
print("=" * 60)

# 1. Check PyTorch CUDA availability
print("\n1. PyTorch CUDA Check:")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU device: {torch.cuda.get_device_name(0)}")
else:
    print("   ⚠️  CUDA is not available. CUDA kernels cannot be used.")

# 2. Check causal_conv1d
print("\n2. causal_conv1d Check:")
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    print("   ✅ causal_conv1d imported successfully")
    
    # Try to import CUDA module
    try:
        import causal_conv1d_cuda
        print("   ✅ causal_conv1d_cuda imported successfully")
        
        # Test if it's actually working
        if torch.cuda.is_available():
            try:
                x = torch.randn(2, 4, 10, device='cuda')
                # weight shape: (dim, width) where width is kernel size (2-4)
                weight = torch.randn(4, 3, device='cuda')
                bias = torch.randn(4, device='cuda')
                out = causal_conv1d_cuda.causal_conv1d_fwd(x, weight, bias, None, True)
                print(f"   ✅ causal_conv1d_cuda test passed (output shape: {out.shape})")
            except Exception as e:
                print(f"   ❌ causal_conv1d_cuda test failed: {e}")
        else:
            print("   ⚠️  Cannot test CUDA kernel (CUDA not available)")
    except ImportError as e:
        print(f"   ❌ causal_conv1d_cuda import failed: {e}")
        print("   ⚠️  CUDA kernel not available, will use fallback")
        
except (ImportError, TypeError) as e:
    error_msg = str(e)
    if "unsupported operand type(s) for |" in error_msg or "TypeError" in str(type(e).__name__):
        print(f"   ⚠️  causal_conv1d import failed due to Python version compatibility")
        print(f"   Error: {error_msg}")
        print("   💡 This is likely due to Python 3.9 incompatibility with causal_conv1d")
        print("   💡 The package uses Python 3.10+ type hints (| operator)")
        print("   💡 Options:")
        print("      1. Upgrade to Python 3.10+")
        print("      2. Use older version: pip install 'causal-conv1d<1.2.0'")
        print("      3. Continue without CUDA kernels (will use fallback)")
    else:
        print(f"   ❌ causal_conv1d import failed: {e}")
        print("   ⚠️  Please install: pip install causal-conv1d>=1.2.0")
except Exception as e:
    print(f"   ❌ Unexpected error importing causal_conv1d: {e}")
    import traceback
    traceback.print_exc()

# 3. Check selective_scan_cuda
print("\n3. selective_scan_cuda Check:")
try:
    import selective_scan_cuda
    print("   ✅ selective_scan_cuda imported successfully")
    
    # Test if it's actually working
    if torch.cuda.is_available():
        try:
            batch, dim, seqlen = 2, 4, 10
            d_state = 4
            
            u = torch.randn(batch, dim, seqlen, device='cuda')
            delta = torch.randn(batch, dim, seqlen, device='cuda')
            A = torch.randn(dim, d_state, device='cuda')
            B = torch.randn(batch, 1, d_state, seqlen, device='cuda')
            C = torch.randn(batch, 1, d_state, seqlen, device='cuda')
            D = torch.randn(dim, device='cuda')
            
            out, x, *rest = selective_scan_cuda.fwd(
                u, delta, A, B, C, D, None, None, False
            )
            print(f"   ✅ selective_scan_cuda test passed (output shape: {out.shape})")
        except Exception as e:
            print(f"   ❌ selective_scan_cuda test failed: {e}")
    else:
        print("   ⚠️  Cannot test CUDA kernel (CUDA not available)")
        
except (ImportError, OSError) as e:
    error_msg = str(e)
    if "undefined symbol" in error_msg or "_ZN3c10" in error_msg:
        print(f"   ❌ selective_scan_cuda import failed: {error_msg}")
        print("   ⚠️  This is a PyTorch version mismatch error!")
        print("   💡 The CUDA kernel was compiled with a different PyTorch version")
        print("   💡 Solutions:")
        print("      1. Reinstall selective_scan_cuda with current PyTorch:")
        print("         pip uninstall selective-scan-cuda")
        print("         pip install selective-scan-cuda --no-cache-dir")
        print("      2. Or reinstall mamba-ssm:")
        print("         pip uninstall mamba-ssm")
        print("         pip install mamba-ssm --no-cache-dir")
        print("      3. Check PyTorch version:")
        print(f"         Current PyTorch: {torch.__version__}")
        print("      4. Continue without CUDA kernels (will use fallback)")
    else:
        print(f"   ❌ selective_scan_cuda import failed: {e}")
        print("   ⚠️  CUDA kernel not available")
        print("   ⚠️  This is required for fast Mamba operations")
except Exception as e:
    print(f"   ❌ Unexpected error importing selective_scan_cuda: {e}")
    import traceback
    traceback.print_exc()

# 4. Check selective_scan_interface
print("\n4. selective_scan_interface Check:")
try:
    from espnet2.asr.state_spaces.selective_scan_interface import (
        selective_scan_fn,
        mamba_inner_fn,
        bimamba_inner_fn,
        mamba_inner_fn_no_out_proj,
    )
    print("   ✅ selective_scan_interface imported successfully")
    print(f"   ✅ selective_scan_fn: {selective_scan_fn is not None}")
    print(f"   ✅ mamba_inner_fn: {mamba_inner_fn is not None}")
    print(f"   ✅ bimamba_inner_fn: {bimamba_inner_fn is not None}")
    print(f"   ✅ mamba_inner_fn_no_out_proj: {mamba_inner_fn_no_out_proj is not None}")
except ImportError as e:
    print(f"   ❌ selective_scan_interface import failed: {e}")

# 5. Check BiMamba module
print("\n5. BiMamba Module Check:")
try:
    from espnet2.asr.state_spaces.bimamba import Mamba as BiMamba
    print("   ✅ BiMamba imported successfully")
    
    # Try to create a small instance
    if torch.cuda.is_available():
        try:
            model = BiMamba(
                d_model=64,
                d_state=16,
                d_conv=4,
                expand=2,
                bimamba_type="v2",
                device='cuda'
            )
            print("   ✅ BiMamba model created successfully")
            
            # Test forward pass
            x = torch.randn(2, 10, 64, device='cuda')
            out = model(x)
            print(f"   ✅ BiMamba forward pass successful (output shape: {out.shape})")
        except Exception as e:
            print(f"   ❌ BiMamba test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("   ⚠️  Cannot test BiMamba (CUDA not available)")
        
except ImportError as e:
    print(f"   ❌ BiMamba import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test Summary:")
print("=" * 60)
if torch.cuda.is_available():
    print("✅ CUDA is available")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA version: {torch.version.cuda}")
else:
    print("❌ CUDA is not available - CUDA kernels cannot be used")

print("\nNote:")
print("- If CUDA kernels are not available, Mamba will use slower fallback implementations")
print("- For best performance, install CUDA kernels:")
print("  pip install causal-conv1d>=1.2.0")
print("  pip install mamba-ssm")
print("\n⚠️  If you see 'undefined symbol' errors:")
print("   This means CUDA kernels were compiled with different PyTorch version")
print("   Solution: Reinstall the packages to recompile with current PyTorch")
print("=" * 60)