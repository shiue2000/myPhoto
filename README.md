# Image Restoration & Colorization Web App

A lightweight Flask-based web app for simple image restoration:
✅ Inpainting using masks  
✅ Colorization of grayscale or damaged images  

## Features
- Efficient memory usage (under 512MB)
- Uses OpenCV DNN colorization (Caffe model)
- Flask web interface for upload + display

## Deployment
### Render
- Connect your GitHub repo
- Choose Docker deployment
- Render uses `render.yaml` + `Dockerfile` to build

### Local run
```bash
docker build -t image-restoration-app .
docker run -p 5000:5000 image-restoration-app
