# BLIP Image Captioning API Service
# ==================================
# Flask API for BLIP model inference

from flask import Flask, request, jsonify
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
)
from PIL import Image
import io
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load BLIP models on startup
logger.info("Loading BLIP models...")
caption_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
caption_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
logger.info("BLIP models loaded successfully!")


def decode_image(image_data: str) -> Image.Image:
    """Decode base64 image data to PIL Image."""
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model": "BLIP"})


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Analyze image and return caption or answer.

    Request JSON:
        - image: base64 encoded image
        - question: (optional) question for VQA mode

    Response JSON:
        - caption: image caption (if no question)
        - answer: answer to question (if question provided)
    """
    try:
        data = request.json

        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        image = decode_image(data["image"])
        question = data.get("question")

        if question:
            # VQA mode
            inputs = vqa_processor(image, question, return_tensors="pt", padding=True)
            outputs = vqa_model.generate(**inputs, max_new_tokens=50)
            answer = vqa_processor.decode(outputs[0], skip_special_tokens=True)

            logger.info(f"VQA - Question: {question}, Answer: {answer}")
            return jsonify({"answer": answer})
        else:
            # Caption mode
            inputs = caption_processor(image, return_tensors="pt", padding=True)
            outputs = caption_model.generate(**inputs, max_new_tokens=50)
            caption = caption_processor.decode(outputs[0], skip_special_tokens=True)

            logger.info(f"Caption: {caption}")
            return jsonify({"caption": caption})

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
