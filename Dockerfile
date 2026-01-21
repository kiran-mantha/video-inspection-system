FROM python:3.10-slim
# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
# Set Hugging Face cache inside container
ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models
ENV TORCH_HOME=/models
WORKDIR /app
# System deps
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*
# Python deps
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Download BLIP model at build time (faster startup)
RUN python -c "from transformers import BlipProcessor, BlipForConditionalGeneration; \
    BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-large'); \
    BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-large'); \
    print('BLIP model downloaded')"
# Copy API code
COPY blip_api.py /app/
# Expose API port
EXPOSE 5007
# Run Flask API
CMD ["python", "blip_api.py"]