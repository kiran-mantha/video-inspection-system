# Ollama Docker Container

This directory contains the Dockerfile for the Ollama container used in video inspection.

## Model Included

- **LLaVA 13B** - Vision-language model for image analysis and structured output

## Build & Run

```bash
# Build the image
docker build -t ollama-llava docker/ollama/

# Run the container
docker run -d -p 11434:11434 --name ollama-llava ollama-llava
```

The model will be pulled during image build (~10-15 minutes for first build).

## Verify Model

```bash
docker exec ollama-llava ollama list
```

Should show:
- `llava:13b`

## Configuration

Set the following environment variables to use the local model:

```bash
set USE_LOCAL_MODELS=true
set OLLAMA_IP=localhost
set OLLAMA_PORT=11434
```
