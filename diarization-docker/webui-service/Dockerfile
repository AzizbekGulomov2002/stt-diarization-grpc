FROM python:3.10-alpine AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Alpine package manager is apk, not apt
RUN apk update && apk add --no-cache \
    gcc \
    musl-dev \
    libffi-dev \
    build-base \
    git

# Install Python packages to /deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --target=/deps -r requirements.txt

# ---------- Final stage ----------
FROM python:3.10-alpine

RUN apk update && apk add --no-cache \
    libffi \
    git

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /deps /usr/local/lib/python3.10/site-packages

# Copy application code
COPY . .

EXPOSE 7860
CMD ["python","app.py"]
