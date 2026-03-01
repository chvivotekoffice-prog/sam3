"""
Example client for SAM3 Multi-Label API
Demonstrates how to use the lightweight inference endpoint
"""

import requests
import base64
from pathlib import Path


def encode_image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded}"


def predict_objects(image_path: str, labels: list, conf_threshold: float = 0.5):
    """
    Send prediction request to SAM3 API
    
    Args:
        image_path: Path to the image file
        labels: List of object labels to detect (e.g., ["door", "carpet"])
        conf_threshold: Confidence threshold (0.0-1.0)
    
    Returns:
        API response as dictionary
    """
    api_url = "http://localhost:5588/predict"
    
    # Encode image
    base64_image = encode_image_to_base64(image_path)
    
    # Prepare request
    payload = {
        "image": base64_image,
        "text": labels,
        "conf_threshold": conf_threshold
    }
    
    # Send request
    print(f"🔍 Detecting: {', '.join(labels)}")
    print(f"📊 Confidence threshold: {conf_threshold}")
    
    response = requests.post(api_url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        if result["status"] == "success":
            print(f"✅ Found {result['total_objects']} objects")
            return result
        else:
            print(f"❌ Error: {result.get('message', 'Unknown error')}")
            return None
    else:
        print(f"❌ HTTP Error: {response.status_code}")
        return None


def main():
    """Example usage"""
    
    # Example 1: Detect doors and carpets
    print("\n" + "="*50)
    print("Example 1: Multi-label detection")
    print("="*50)
    
    result = predict_objects(
        image_path="your_image.jpg",
        labels=["door", "carpet"],
        conf_threshold=0.5
    )
    
    if result:
        for i, obj in enumerate(result["result"]):
            print(f"\n[{i+1}] {obj['text'].upper()}")
            print(f"    Score: {obj['score']:.3f}")
            print(f"    Bounding Box: {obj['box']}")
            print(f"    Polygon Points: {len(obj['polygon'][0]['points'])} points")
    
    # Example 2: Higher confidence threshold
    print("\n" + "="*50)
    print("Example 2: Higher confidence")
    print("="*50)
    
    result = predict_objects(
        image_path="your_image.jpg",
        labels=["window", "table"],
        conf_threshold=0.7
    )


if __name__ == "__main__":
    main()
