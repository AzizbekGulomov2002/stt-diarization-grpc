# stt-diarization-grpc

STT and speaker diarization stack built on **gRPC**. The project runs multiple services together with **Docker** and is significantly faster in practice than a legacy sequential **1x1** pipeline.

## What's included

- `diarization-docker/stt-service` – speech-to-text service  
- `diarization-docker/diarization-service` – diarization (MSDD, gender, gRPC)  
- `diarization-docker/webui-service` – web UI  
- `diarization-docker/docker-compose.yml` – container orchestration  

## Performance (10-minute audio)

| Approach | Time |
| --- | --- |
| Legacy 1x1 pipeline | ~10 minutes |
| gRPC pipeline | ~9 seconds |

The main drivers are optimized inter-service communication, better use of parallelism/streaming, and stable deployment in Docker.

## Getting started

1. Copy the environment template:

   ```bash
   cp diarization-docker/.env.example diarization-docker/.env
   ```

2. Fill in your values in `.env`. **Do not commit secrets** (tokens, API keys).

3. Start the stack:

   ```bash
   docker compose -f diarization-docker/docker-compose.yml up --build
   ```

More detail: see `diarization-docker/README.md`.
