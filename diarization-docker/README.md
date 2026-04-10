# STT Diarization gRPC (Docker)

Bu loyiha speaker diarization va STT pipeline ni Docker konteynerlarda ishga tushiradi. gRPC asosidagi yangi oqim orqali eski 1x1 ishlov berish usuliga nisbatan ancha tez natija beradi.

## Servislar

- `stt-service` - nutqni matnga o'girish (STT)
- `diarization-service` - speaker diarization va gender aniqlash
- `webui-service` - foydalanuvchi uchun web interfeys
- `docker-compose.yml` - barcha servislarni birga ko'tarish uchun konfiguratsiya

## Performance taqqoslash

10 minutlik audio fayl ustidagi real taqqoslash:

| Pipeline | Vaqt |
| --- | --- |
| Eski 1x1 (ketma-ket) | ~10 minut |
| gRPC (optimallashtirilgan) | ~9 sekund |

gRPC bilan servislar orasidagi uzatish tezlashadi, navbatlash kamayadi va umumiy throughput oshadi.

## Loyiha tuzilmasi

```text
.
├── docker-compose.yml
├── .env.example
├── stt-service/
├── diarization-service/
└── webui-service/
```

## .env (maxfiy kalitlar yashirilgan)

`.env` faylni commit qilmang. Avval `.env.example` dan nusxa oling:

```bash
cp .env.example .env
```

Namuna:

```env
NVIDIA_VISIBLE_DEVICES=0
HF_TOKEN=YOUR_HUGGINGFACE_TOKEN
STT_MODEL=YOUR_STT_MODEL_ID
GENDER_MODEL=YOUR_GENDER_MODEL_ID
MSDD_MODEL=YOUR_MSDD_MODEL_ID
STT_API=http://stt-service:8000/transcribe/
```

## Ishga tushirish

```bash
docker compose up --build
```

## Foydali buyruqlar

To'xtatish va tozalash:

```bash
docker compose down --volumes --remove-orphans
```

Loglarni ko'rish:

```bash
docker compose logs -f
```
