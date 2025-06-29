FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV FLASK_ENV=production

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1-mesa-glx \
        libopencv-dev \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app/

RUN ls -lh /app/model && sha256sum /app/model/colorization_release_v2.caffemodel || true
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt || \
    pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]