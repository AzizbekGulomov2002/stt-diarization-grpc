# STT Diarization gRPC (Docker)

Speaker diarization and STT run in Docker containers. The **gRPC**-based path replaces a slow legacy **1x1** (fully sequential) workflow with a much faster pipeline.

## Services

- `stt-service` – speech-to-text  
- `diarization-service` – speaker diarization and gender estimation  
- `webui-service` – simple web UI  
- `docker-compose.yml` – runs all services together  

## Performance comparison

On a **10-minute** audio sample:

| Pipeline | Time |
| --- | --- |
| Legacy 1x1 (sequential) | ~10 minutes |
| Optimized gRPC | ~9 seconds |

gRPC reduces overhead between services, cuts queueing, and improves end-to-end throughput.

## Layout

```text
.
├── docker-compose.yml
├── .env.example
├── stt-service/
├── diarization-service/
└── webui-service/
```

## Environment variables

Do **not** commit `.env`. Start from the example file:

```bash
cp .env.example .env
```

Template (replace placeholders with your real values locally only):

```env
NVIDIA_VISIBLE_DEVICES=0
HF_TOKEN=YOUR_HUGGINGFACE_TOKEN
STT_MODEL=YOUR_STT_MODEL_ID
GENDER_MODEL=YOUR_GENDER_MODEL_ID
MSDD_MODEL=YOUR_MSDD_MODEL_ID
STT_API=http://stt-service:8000/transcribe/
```

## Run

```bash
docker compose up --build
```

## Useful commands

Stop and clean up:

```bash
docker compose down --volumes --remove-orphans
```

Logs:

```bash
docker compose logs -f
```
