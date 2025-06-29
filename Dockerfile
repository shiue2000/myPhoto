FROM python:3.9-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required by OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy app files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
