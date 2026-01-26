# README for Docker Setup

## Quick Start

### 1. Build and start services:
```bash
cd docker
docker-compose up -d --build
```

### 2. Pull Mistral model (first time only):
```bash
docker exec video-inspection-mistral ollama pull mistral
```

### 3. Verify services are running:
```bash
# Check BLIP
curl http://localhost:5050/health

# Check Ollama/Mistral
curl http://localhost:11434/api/tags
```

### 4. Stop services:
```bash
docker-compose down
```

## GPU Support

If you have an NVIDIA GPU, uncomment the GPU sections in `docker-compose.yml` for significantly faster inference.

## Troubleshooting

- **BLIP slow on first request**: The model is loaded into memory on first request
- **Mistral model not found**: Run `docker exec video-inspection-mistral ollama pull mistral`
- **Port conflicts**: Modify ports in `docker-compose.yml` and update `config.py` accordingly
