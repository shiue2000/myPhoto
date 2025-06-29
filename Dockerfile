# First stage: build dependencies
FROM python:3.9-slim as builder

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required by OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1 \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Pre-install Python dependencies into a virtualenv-like folder
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# Second stage: runtime-only image
FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install only runtime system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy the app code
COPY . /app

EXPOSE 5000

# Add basic healthcheck (adjust as needed for your app)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD curl -f http://localhost:5000/ || exit 1

CMD ["python", "app.py"]
