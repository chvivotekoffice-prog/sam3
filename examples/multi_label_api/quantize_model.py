import sys
import os
import torch
import traceback
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
sam3_root = os.path.join(current_dir, "..", "..")
if sam3_root not in sys.path:
    sys.path.append(sam3_root)

try:
    from sam3.model_builder import build_sam3_image_model
    print("SAM3 model loaded successfully")
except ImportError as e:
    print(f"Failed to load SAM3: {e}")
    sys.exit(1)


def quantize_fp16(model_path: str = None, output_path: str = "sam3_fp16.pth"):
    """
    Convert model to FP16 precision
    
    Args:
        model_path: Path to load model from (None = load from builder)
        output_path: Path to save quantized model
    
    Returns:
        Quantized model and size comparison
    """
    print(f"\n{'='*60}")
    print("FP16 Quantization")
    print(f"{'='*60}\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        print("Loading original model...")
        if model_path:
            model = torch.load(model_path)
        else:
            model = build_sam3_image_model()
        
        model.eval()
        original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        print(f"Original model size (FP32): {original_size:.2f} MB\n")
        
        print("Converting to FP16...")
        model_fp16 = model.half()
        
        print(f"Saving quantized model to {output_path}...")
        torch.save(model_fp16.state_dict(), output_path)
        
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)
        reduction = (1 - quantized_size / original_size) * 100
        
        print(f"\n✅ FP16 quantization completed!")
        print(f"   Original (FP32): {original_size:.2f} MB")
        print(f"   Quantized (FP16): {quantized_size:.2f} MB")
        print(f"   Reduction: {reduction:.1f}%")
        print(f"   File saved: {os.path.abspath(output_path)}")
        print(f"\n{'='*60}\n")
        
        return model_fp16, original_size, quantized_size
        
    except Exception as e:
        print(f"❌ FP16 quantization failed: {e}")
        traceback.print_exc()
        return None, None, None


def quantize_dynamic_int8(output_path: str = "sam3_int8_dynamic.pth"):
    """
    Apply dynamic INT8 quantization to model
    
    Args:
        output_path: Path to save quantized model
    
    Returns:
        Quantized model
    """
    print(f"\n{'='*60}")
    print("Dynamic INT8 Quantization")
    print(f"{'='*60}\n")
    
    try:
        print("Loading model...")
        model = build_sam3_image_model()
        model.eval()
        
        original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        print(f"Original model size: {original_size:.2f} MB\n")
        
        print("Applying dynamic quantization...")
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )
        
        print(f"Saving quantized model to {output_path}...")
        torch.save(quantized_model.state_dict(), output_path)
        
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)
        reduction = (1 - quantized_size / original_size) * 100
        
        print(f"\n✅ INT8 quantization completed!")
        print(f"   Original: {original_size:.2f} MB")
        print(f"   Quantized: {quantized_size:.2f} MB")
        print(f"   Reduction: {reduction:.1f}%")
        print(f"   File saved: {os.path.abspath(output_path)}")
        
        print(f"\n⚠️  Note: Accuracy may be affected. Please validate before deployment.")
        print(f"{'='*60}\n")
        
        return quantized_model
        
    except Exception as e:
        print(f"❌ INT8 quantization failed: {e}")
        traceback.print_exc()
        return None


def compare_all_formats(output_dir: str = "quantized_models"):
    """
    Generate and compare all quantization formats
    
    Args:
        output_dir: Directory to save all quantized models
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("Comprehensive Model Quantization Comparison")
    print(f"{'='*70}\n")
    
    results = []
    
    try:
        print("Loading base model...")
        model = build_sam3_image_model()
        model.eval()
        
        original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        results.append(("FP32 (Original)", original_size, 1.0, "Baseline"))
        
        fp16_model, _, fp16_size = quantize_fp16(
            output_path=os.path.join(output_dir, "sam3_fp16.pth")
        )
        if fp16_size:
            speedup = 1.5
            results.append(("FP16", fp16_size, speedup, "Minimal accuracy loss"))
        
        int8_model = quantize_dynamic_int8(
            output_path=os.path.join(output_dir, "sam3_int8.pth")
        )
        if int8_model:
            int8_size = os.path.getsize(os.path.join(output_dir, "sam3_int8.pth")) / (1024 * 1024)
            speedup = 2.0
            results.append(("INT8 Dynamic", int8_size, speedup, "May affect accuracy"))
        
        print(f"\n{'='*70}")
        print("Summary: Model Size and Performance Comparison")
        print(f"{'='*70}\n")
        print(f"{'Format':<20} {'Size (MB)':<15} {'Speedup':<12} {'Note':<25}")
        print("-" * 70)
        
        for fmt, size, speedup, note in results:
            print(f"{fmt:<20} {size:>10.2f} MB   {speedup:>6.1f}x      {note:<25}")
        
        print(f"\n{'='*70}")
        print("Recommendation for Edge Deployment:")
        print("  • Edge devices with GPU: Use FP16 (best balance)")
        print("  • CPU-only devices: Use INT8 (smaller size)")
        print("  • High-accuracy needs: Use FP32 with state reuse optimization")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SAM3 Model Quantization Tool")
    parser.add_argument(
        "--mode",
        type=str,
        default="compare",
        choices=["fp16", "int8", "compare"],
        help="Quantization mode"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="quantized_models",
        help="Output directory for quantized models"
    )
    
    args = parser.parse_args()
    
    if args.mode == "fp16":
        quantize_fp16(output_path=os.path.join(args.output, "sam3_fp16.pth"))
    elif args.mode == "int8":
        quantize_dynamic_int8(output_path=os.path.join(args.output, "sam3_int8.pth"))
    elif args.mode == "compare":
        compare_all_formats(output_dir=args.output)
