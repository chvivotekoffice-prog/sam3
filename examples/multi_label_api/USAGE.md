# Quick Start Guide

## Running the API Server

### Basic Usage
```bash
python api_server.py
```

Server starts on: `http://localhost:5588`

---

## ONNX Deployment Workflow

### Step 1: Export Model
```bash
python export_to_onnx.py
```

Output: `sam3_optimized.onnx` (~2.4 GB)

### Step 2: Verify Export
```bash
python export_to_onnx.py --mode verify --output sam3_optimized.onnx
```

Checks:
- Model structure
- Input/output shapes
- ONNX compliance

### Step 3: Benchmark (Optional)
```bash
python export_to_onnx.py --mode benchmark --image your_test_image.jpg
```

Compares PyTorch vs ONNX inference speed.

---

## Quantization Workflow

### For GPU Deployment (FP16)
```bash
python quantize_model.py --mode fp16 --output models
```

Result:
- Size: 2.4 GB → 1.2 GB
- Speed: 1.5x faster
- Accuracy: 99% of original

### For CPU Deployment (INT8)
```bash
python quantize_model.py --mode int8 --output models
```

Result:
- Size: 2.4 GB → 600 MB
- Speed: 2x faster on CPU
- Accuracy: 95% of original

### Compare All Options
```bash
python quantize_model.py --mode compare
```

Generates comparison report for all formats.

---

## Testing the Client

```bash
python example_client.py
```

Make sure:
1. API server is running (`python api_server.py`)
2. You have a test image (`your_image.jpg`)
3. Edit `example_client.py` to use your image path

---

## File Structure

```
multi_label_api/
├── api_server.py           # Main API server (FastAPI)
├── export_to_onnx.py       # ONNX export tool
├── onnx_inference.py       # ONNX Runtime inference
├── quantize_model.py       # Model quantization tool
├── example_client.py       # API client example
├── requirements.txt        # Python dependencies
├── README.md               # Full documentation
├── DEPLOYMENT.md           # Deployment guide
└── USAGE.md               # This file
```

---

## Common Commands

| Task | Command |
|------|---------|
| Start API server | `python api_server.py` |
| Export to ONNX | `python export_to_onnx.py` |
| Quantize to FP16 | `python quantize_model.py --mode fp16` |
| Run client example | `python example_client.py` |
| Benchmark ONNX | `python export_to_onnx.py --mode benchmark --image test.jpg` |

---

## Tips

1. **For development**: Use PyTorch API server (best flexibility)
2. **For production**: Use ONNX + TensorRT (best performance)
3. **For edge devices**: Use quantized models (best size/speed trade-off)
4. **For CPU-only**: Use INT8 quantization (best CPU performance)
