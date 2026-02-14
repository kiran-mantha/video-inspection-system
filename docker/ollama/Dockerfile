# Ollama Container with LLaVA
# ============================
# Docker container for Ollama with LLaVA vision model
# Model is pulled during build for faster startup and consistency

FROM ollama/ollama:latest

# Expose Ollama API port
EXPOSE 11434

# Start Ollama service in background and pull LLaVA during build
# This ensures the model is baked into the image for faster container startup
RUN ollama serve & \
    sleep 5 && \
    ollama pull llava:13b && \
    pkill ollama

# The ollama/ollama image handles the entrypoint automatically
# Container will start with: ollama serve
