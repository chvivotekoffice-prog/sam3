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


def export_sam3_to_onnx(
    output_path: str = "sam3_optimized.onnx",
    device: str = "cuda",
    opset_version: int = 17,
    dynamic_batch: bool = True
):
    """
    Export SAM3 model to ONNX format for edge deployment
    
    Args:
        output_path: Path to save the ONNX model
        device: Device to use for export (cuda/cpu)
        opset_version: ONNX opset version (17 recommended for best compatibility)
        dynamic_batch: Whether to support dynamic batch size
    """
    print(f"\n{'='*60}")
    print("SAM3 to ONNX Export Tool")
    print(f"{'='*60}\n")
    
    device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
    print(f"Using device: {device}")
    
    try:
        print("Loading SAM3 model...")
        model = build_sam3_image_model().to(device)
        model.eval()
        model.float()
        print("Model loaded successfully\n")
        
        print("Preparing dummy inputs for export...")
        dummy_image = torch.randn(1, 3, 1024, 1024, device=device)
        
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                'image': {0: 'batch_size', 2: 'height', 3: 'width'},
                'masks': {0: 'batch_size'},
                'scores': {0: 'batch_size'}
            }
        
        print(f"Exporting to ONNX (opset={opset_version})...")
        print(f"Output path: {output_path}")
        
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_image,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['image'],
                output_names=['masks', 'scores'],
                dynamic_axes=dynamic_axes,
                verbose=False
            )
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n✅ Export successful!")
        print(f"   Model size: {file_size:.2f} MB")
        print(f"   Path: {os.path.abspath(output_path)}")
        
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print(f"   Validation: Passed")
        except ImportError:
            print(f"   Validation: Skipped (onnx package not installed)")
        except Exception as e:
            print(f"   Validation: Warning - {e}")
        
        print(f"\n{'='*60}")
        print("Export completed successfully!")
        print(f"{'='*60}\n")
        
        print("Next steps:")
        print("1. Test with ONNX Runtime: python onnx_inference.py")
        print("2. Optimize for TensorRT: trtexec --onnx=sam3_optimized.onnx")
        print("3. Deploy to edge devices\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        traceback.print_exc()
        return False


def export_with_quantization(output_dir: str = "models"):
    """
    Export SAM3 with different quantization options
    """
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*60}")
    print("SAM3 Quantized Export")
    print(f"{'='*60}\n")
    
    try:
        print("Loading SAM3 model...")
        model = build_sam3_image_model().to(device)
        model.eval()
        
        print("\n1. Exporting FP32 model...")
        model.float()
        export_sam3_to_onnx(
            output_path=os.path.join(output_dir, "sam3_fp32.onnx"),
            device=device
        )
        
        if device == "cuda":
            print("\n2. Exporting FP16 model...")
            model.half()
            export_sam3_to_onnx(
                output_path=os.path.join(output_dir, "sam3_fp16.onnx"),
                device=device
            )
        
        print("\n" + "="*60)
        print("Model Size Comparison:")
        print("="*60)
        
        for filename in os.listdir(output_dir):
            if filename.endswith('.onnx'):
                filepath = os.path.join(output_dir, filename)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  {filename:20s}: {size_mb:>8.2f} MB")
        
        print(f"\n{'='*60}\n")
        
    except Exception as e:
        print(f"❌ Quantization export failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export SAM3 to ONNX format")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="sam3_optimized.onnx",
        help="Output ONNX file path"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for export"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Export with multiple quantization options"
    )
    
    args = parser.parse_args()
    
    if args.quantize:
        export_with_quantization()
    else:
        export_sam3_to_onnx(
            output_path=args.output,
            device=args.device,
            opset_version=args.opset
        )
