# stt-diarization-grpc

gRPC asosida qurilgan STT + speaker diarization stack. Loyiha Docker orqali bir necha servisni birgalikda ishga tushiradi va real ishlashda oldingi 1x1 pipeline bilan solishtirganda sezilarli tezlashuv beradi.

## Nima bor

- `diarization-docker/stt-service` - STT xizmati
- `diarization-docker/diarization-service` - diarization (MSDD, gender, gRPC)
- `diarization-docker/webui-service` - web interfeys
- `diarization-docker/docker-compose.yml` - konteynerlarni orkestratsiya qilish

## Performance taqqoslash (10 minut audio)

| Yondashuv | Natija vaqti |
| --- | --- |
| Eski 1x1 pipeline | ~10 minut |
| gRPC pipeline | ~9 sekund |

Bu o'zgarishning asosiy sababi - servislar orasidagi aloqani optimallashtirish, oqimni parallel ishlatish va Docker ichida barqaror deploy qilish.

## Ishga tushirish

1. `diarization-docker/.env.example` faylidan nusxa oling:
   - `cp diarization-docker/.env.example diarization-docker/.env`
2. Kerakli qiymatlarni `.env`ga kiriting (tokenlarni repoga commit qilmang).
3. Servislarni ishga tushiring:
   - `docker compose -f diarization-docker/docker-compose.yml up --build`