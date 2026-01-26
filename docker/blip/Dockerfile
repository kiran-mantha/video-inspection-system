# BLIP Image Captioning Service
# ==============================
# Docker container for BLIP model inference

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    flask==3.0.0 \
    transformers==4.36.0 \
    torch==2.1.0 \
    pillow==10.0.0 \
    accelerate==0.25.0 \
    "numpy<2"

# Copy application code
COPY api.py .

# Expose port
EXPOSE 5050

# Download models on build (optional - can also download at runtime)
RUN python -c "from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering; \
    BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base'); \
    BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base'); \
    BlipProcessor.from_pretrained('Salesforce/blip-vqa-base'); \
    BlipForQuestionAnswering.from_pretrained('Salesforce/blip-vqa-base')"

# Run the API
CMD ["python", "api.py"]
