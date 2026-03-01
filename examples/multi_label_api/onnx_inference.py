import onnxruntime as ort
import numpy as np
import torch
import cv2
import time
from PIL import Image
from typing import List, Tuple
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sam3_root = os.path.join(current_dir, "..", "..")
if sam3_root not in sys.path:
    sys.path.append(sam3_root)


class ONNXInferenceEngine:
    """
    ONNX Runtime inference engine for SAM3
    Optimized for edge deployment
    """
    
    def __init__(self, onnx_path: str, device: str = "cuda"):
        """
        Initialize ONNX inference engine
        
        Args:
            onnx_path: Path to ONNX model file
            device: Device to use (cuda/cpu)
        """
        print(f"Initializing ONNX Runtime on {device}...")
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(
            onnx_path,
            providers=providers
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"✅ ONNX model loaded successfully")
        print(f"   Input: {self.input_name}")
        print(f"   Outputs: {self.output_names}")
        print(f"   Provider: {self.session.get_providers()[0]}\n")
    
    def preprocess_image(self, image: Image.Image, size: Tuple[int, int] = (1024, 1024)) -> np.ndarray:
        """
        Preprocess image for ONNX model input
        
        Args:
            image: PIL Image
            size: Target size (height, width)
        
        Returns:
            Preprocessed image tensor
        """
        image = image.resize(size, Image.BILINEAR)
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = image_np.transpose(2, 0, 1)
        image_np = np.expand_dims(image_np, axis=0)
        return image_np
    
    def inference(self, image: Image.Image) -> dict:
        """
        Run inference on image
        
        Args:
            image: PIL Image
        
        Returns:
            Dictionary with masks and scores
        """
        image_tensor = self.preprocess_image(image)
        
        outputs = self.session.run(
            self.output_names,
            {self.input_name: image_tensor}
        )
        
        return {
            "masks": outputs[0],
            "scores": outputs[1]
        }


def benchmark_onnx_vs_pytorch(image_path: str, onnx_model_path: str, num_iterations: int = 10):
    """
    Compare performance between ONNX and PyTorch
    
    Args:
        image_path: Path to test image
        onnx_model_path: Path to ONNX model
        num_iterations: Number of iterations for benchmarking
    """
    print(f"\n{'='*60}")
    print("Performance Benchmark: ONNX vs PyTorch")
    print(f"{'='*60}\n")
    
    image = Image.open(image_path).convert("RGB")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Test image: {image_path}")
    print(f"Image size: {image.size}")
    print(f"Iterations: {num_iterations}\n")
    
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        
        print("1. Testing PyTorch inference...")
        model = build_sam3_image_model().to(device)
        processor = Sam3Processor(model)
        model.eval()
        
        pytorch_times = []
        for i in range(num_iterations):
            start = time.time()
            inference_state = processor.set_image(image)
            output = processor.set_text_prompt(state=inference_state, prompt="object")
            torch.cuda.synchronize() if device == "cuda" else None
            elapsed = time.time() - start
            pytorch_times.append(elapsed)
            print(f"   Iteration {i+1}/{num_iterations}: {elapsed:.3f}s")
        
        avg_pytorch = sum(pytorch_times) / len(pytorch_times)
        print(f"   Average: {avg_pytorch:.3f}s\n")
        
    except Exception as e:
        print(f"   PyTorch test failed: {e}\n")
        avg_pytorch = None
    
    print("2. Testing ONNX inference...")
    try:
        engine = ONNXInferenceEngine(onnx_model_path, device)
        
        onnx_times = []
        for i in range(num_iterations):
            start = time.time()
            output = engine.inference(image)
            elapsed = time.time() - start
            onnx_times.append(elapsed)
            print(f"   Iteration {i+1}/{num_iterations}: {elapsed:.3f}s")
        
        avg_onnx = sum(onnx_times) / len(onnx_times)
        print(f"   Average: {avg_onnx:.3f}s\n")
        
    except Exception as e:
        print(f"   ONNX test failed: {e}\n")
        avg_onnx = None
    
    print(f"{'='*60}")
    print("Results:")
    print(f"{'='*60}")
    
    if avg_pytorch and avg_onnx:
        speedup = avg_pytorch / avg_onnx
        print(f"PyTorch:  {avg_pytorch:.3f}s")
        print(f"ONNX:     {avg_onnx:.3f}s")
        print(f"Speedup:  {speedup:.2f}x")
        
        if speedup > 1.2:
            print(f"\n✅ ONNX is {speedup:.2f}x faster than PyTorch")
        elif speedup > 0.8:
            print(f"\n⚖️  ONNX and PyTorch have similar performance")
        else:
            print(f"\n⚠️  PyTorch is faster (check ONNX optimization)")
    else:
        if avg_pytorch:
            print(f"PyTorch:  {avg_pytorch:.3f}s")
        if avg_onnx:
            print(f"ONNX:     {avg_onnx:.3f}s")
    
    print(f"\n{'='*60}\n")


def verify_onnx_model(onnx_path: str):
    """
    Verify ONNX model structure and metadata
    
    Args:
        onnx_path: Path to ONNX model file
    """
    try:
        import onnx
        
        print(f"\n{'='*60}")
        print("ONNX Model Verification")
        print(f"{'='*60}\n")
        
        model = onnx.load(onnx_path)
        
        print("Model structure:")
        print(f"  IR version: {model.ir_version}")
        print(f"  Producer: {model.producer_name}")
        print(f"  Opset version: {model.opset_import[0].version}\n")
        
        print("Inputs:")
        for inp in model.graph.input:
            shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in inp.type.tensor_type.shape.dim]
            print(f"  {inp.name}: {shape}")
        
        print("\nOutputs:")
        for out in model.graph.output:
            shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in out.type.tensor_type.shape.dim]
            print(f"  {out.name}: {shape}")
        
        onnx.checker.check_model(model)
        print("\n✅ Model validation passed")
        
        file_size = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"\nModel file size: {file_size:.2f} MB")
        print(f"{'='*60}\n")
        
    except ImportError:
        print("onnx package not installed. Install with: pip install onnx")
    except Exception as e:
        print(f"Verification failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SAM3 ONNX Export and Verification")
    parser.add_argument(
        "--mode",
        type=str,
        default="export",
        choices=["export", "verify", "benchmark"],
        help="Operation mode"
    )
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
        help="Device for export/inference"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Test image path (for benchmark mode)"
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=10,
        help="Number of iterations for benchmark"
    )
    
    args = parser.parse_args()
    
    if args.mode == "export":
        export_sam3_to_onnx(
            output_path=args.output,
            device=args.device
        )
    elif args.mode == "verify":
        verify_onnx_model(args.output)
    elif args.mode == "benchmark":
        if not args.image:
            print("Error: --image is required for benchmark mode")
            sys.exit(1)
        benchmark_onnx_vs_pytorch(
            image_path=args.image,
            onnx_model_path=args.output,
            num_iterations=args.iterations
        )
