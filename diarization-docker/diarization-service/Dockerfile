# --------- Stage 1: Build Stage ---------
FROM python:3.10-slim AS builder

# Install system packages for building
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /install

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --------- Stage 2: Runtime Stage ---------
FROM python:3.10-slim

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    sox \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy application code
COPY . .
# Expose port
EXPOSE 50051

CMD ["python3", "server.py"]