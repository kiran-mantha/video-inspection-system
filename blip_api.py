# BLIP API Server for Docker Container
# =====================================
# Replace infer.py with this file or add as blip_api.py
# Run with: python blip_api.py
#
# Endpoints:
#   POST /analyze - Analyze image (base64 encoded)
#   GET /health   - Health check
from flask import Flask, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io
import base64
import torch

app = Flask(__name__)
# Model configuration
MODEL_ID = "Salesforce/blip-image-captioning-large"
# Load model at startup
print(f"Loading BLIP model: {MODEL_ID}...")
processor = BlipProcessor.from_pretrained(MODEL_ID)
model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)
# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"BLIP model loaded on {device.upper()}")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model": MODEL_ID, "device": device})


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Analyze image and return caption.
    Request JSON:
        {
            "image": "<base64 encoded image data>"
        }
    Response JSON:
        {
            "caption": "a man walking down the street",
            "success": true
        }
    """
    try:
        data = request.json
        if not data or "image" not in data:
            return jsonify({"success": False, "error": "Missing 'image' field"}), 400
        # Decode base64 image
        image_data = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        # Process image
        inputs = processor(image, return_tensors="pt").to(device)
        # Generate caption
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(output[0], skip_special_tokens=True)
        return jsonify({"success": True, "caption": caption})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    # Run on all interfaces, port 5000
    app.run(host="0.0.0.0", port=5000, debug=False)
