import sys
import torch
import cv2
import numpy as np
import io
import base64
import os
import traceback
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
from shapely.geometry import Polygon


class Point(BaseModel):
    x: float
    y: float


class PredictRequest(BaseModel):
    image: str
    text: List[str]
    centers: Optional[List[Point]] = None
    conf_threshold: Optional[float] = 0.5


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    print("SAM3 model loaded successfully")
except ImportError as e:
    print(f"Failed to load SAM3: {e}")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
processor = None

print(f"Initializing SAM3 on {device}...")
try:
    model = build_sam3_image_model().to(device)
    processor = Sam3Processor(model)
    model.eval()
    model.float()
    print("SAM3 initialized successfully")
except Exception as e:
    print(f"Initialization error: {e}")
    traceback.print_exc()


def decode_image(base64_string: str) -> Image.Image:
    if "," in base64_string:
        _, encoded = base64_string.split(",", 1)
    else:
        encoded = base64_string
    image_bytes = base64.b64decode(encoded)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def parse_labels(text_list: List[str]) -> List[str]:
    labels = []
    for text in text_list:
        labels.extend([label.strip() for label in text.split(",") if label.strip()])
    return labels


def simplify_polygon(contour, max_points: int = 8) -> np.ndarray:
    hull = cv2.convexHull(contour)
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    
    if len(approx) > max_points:
        epsilon_multiplier = 2.0
        while len(approx) > max_points and epsilon_multiplier < 10:
            new_epsilon = epsilon * epsilon_multiplier
            approx = cv2.approxPolyDP(hull, new_epsilon, True)
            epsilon_multiplier += 0.5
    
    return approx


def extract_masks_to_candidates(masks, scores, label: str, threshold: float, min_area: float = 800):
    candidates = []
    
    if masks is None or len(masks) == 0:
        return candidates
    
    for i in range(len(masks)):
        score = scores[i].item() if torch.is_tensor(scores) else scores[i]
        if score < threshold:
            continue
        
        mask_np = masks[i].detach().cpu().numpy()
        if mask_np.ndim == 3:
            mask_np = mask_np[0]
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            hull = cv2.convexHull(contour)
            area = cv2.contourArea(hull)
            if area < min_area:
                continue
            
            approx = simplify_polygon(contour)
            if len(approx) < 3:
                continue
            
            points = [{"x": float(p[0][0]), "y": float(p[0][1])} for p in approx]
            x, y, w, h = cv2.boundingRect(approx)
            
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x
            cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y
            
            candidates.append({
                "label": label,
                "points": points,
                "score": score,
                "centroid": {"x": cX, "y": cY},
                "box": {"x": float(x), "y": float(y), "width": float(w), "height": float(h)},
                "polygon": Polygon([[p["x"], p["y"]] for p in points]).buffer(0)
            })
    
    return candidates


def apply_nms_per_label(candidates: List[dict], overlap_threshold: float = 0.8) -> List[dict]:
    sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
    filtered = []
    
    for candidate in sorted_candidates:
        is_redundant = False
        replace_idx = -1
        
        for idx, existing in enumerate(filtered):
            cand_poly = candidate['polygon']
            exist_poly = existing['polygon']
            inter_area = cand_poly.intersection(exist_poly).area
            cand_area = cand_poly.area
            exist_area = exist_poly.area
            
            cand_overlap_ratio = inter_area / cand_area if cand_area > 0 else 0
            exist_overlap_ratio = inter_area / exist_area if exist_area > 0 else 0
            
            if cand_overlap_ratio > overlap_threshold:
                is_redundant = True
                break
            
            if exist_overlap_ratio > overlap_threshold:
                if cand_area > exist_area:
                    replace_idx = idx
                    break
                else:
                    is_redundant = True
                    break
            
            union_area = cand_area + exist_area - inter_area
            iou = inter_area / union_area if union_area > 0 else 0
            if iou > 0.5:
                is_redundant = True
                break
        
        if replace_idx >= 0:
            filtered[replace_idx] = candidate
        elif not is_redundant:
            filtered.append(candidate)
    
    return filtered


def format_results(filtered_candidates: List[dict]) -> List[dict]:
    results = []
    for item in filtered_candidates:
        results.append({
            "score": float(item["score"]),
            "text": item["label"],
            "polygon": [{"points": item["points"]}],
            "box": item["box"],
            "centroid": item["centroid"]
        })
    return results


@app.post("/predict")
async def predict(req: PredictRequest):
    if processor is None:
        return {"status": "error", "message": "Model not initialized"}
    
    try:
        pil_image = decode_image(req.image)
        labels = parse_labels(req.text)
        
        inference_state = processor.set_image(pil_image)
        all_candidates = []
        
        for label in labels:
            output = processor.set_text_prompt(state=inference_state, prompt=label)
            masks = output.get("masks")
            scores = output.get("scores")
            
            candidates = extract_masks_to_candidates(
                masks, scores, label, req.conf_threshold
            )
            all_candidates.extend(candidates)
        
        unique_labels = set(c["label"] for c in all_candidates)
        final_results = []
        
        for label in unique_labels:
            label_candidates = [c for c in all_candidates if c["label"] == label]
            filtered = apply_nms_per_label(label_candidates)
            final_results.extend(format_results(filtered))
        
        return {
            "status": "success",
            "total_objects": len(final_results),
            "result": final_results
        }
    
    except Exception as e:
        print(f"Error: {traceback.format_exc()}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5588)
