import os
import cv2
import numpy as np
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static')

# --- Directory Setup ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, 'uploads')
OUTPUT_FOLDER = os.path.join(STATIC_FOLDER, 'outputs')
MODEL_FOLDER = os.path.join(BASE_DIR, 'model')

os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Model paths ---

protoPath = os.path.join(MODEL_FOLDER, 'colorization_deploy_v2.prototxt')
modelPath = os.path.join(MODEL_FOLDER, 'colorization_release_v2.caffemodel')
hullPath = os.path.join(MODEL_FOLDER, 'pts_in_hull.npy')

assert os.path.exists(protoPath), f"Missing proto file: {protoPath}"
assert os.path.exists(modelPath), f"Missing model file: {modelPath}"
assert os.path.exists(hullPath), f"Missing hull file: {hullPath}"

COLORIZATION_MODEL_AVAILABLE = False
try:
    print("Loading colorization model...")
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    pts_in_hull = np.load(hullPath)

    if pts_in_hull.shape != (313, 2):
        raise ValueError(f"pts_in_hull shape invalid: {pts_in_hull.shape}")

    pts = pts_in_hull.transpose().reshape(2, 313, 1, 1).astype(np.float32)

    net.getLayer(net.getLayerId('class8_ab')).blobs = [pts]
    net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, dtype=np.float32)]

    COLORIZATION_MODEL_AVAILABLE = True
    print("Colorization model loaded successfully.")
except Exception as e:
    print(f"Failed to load colorization model: {e}")

MAX_DIMENSION = 2048

def resize_img(img):
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(h, w)
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def adjust_brightness_contrast(img, brightness=0, contrast=20):
    alpha = 1 + contrast / 100.0
    return cv2.convertScaleAbs(img, alpha=alpha, beta=brightness)

def colorize_image_local(input_path, output_path):
    if not COLORIZATION_MODEL_AVAILABLE:
        print("Colorization model not available.")
        return False
    print(f"Trying to read image: {input_path}")
    img = cv2.imread(input_path)
    if img is None:
        print("ERROR: Could not read image. Is it a valid image file?")
        return False
    else:
        print(f"Image read successfully. Shape: {img.shape}")
    img = resize_img(img)
    h, w = img.shape[:2]

    img_rgb = img.astype("float32") / 255.0
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    L_resized = cv2.resize(L, (224, 224))
    L_resized -= 50  # mean-centering

    try:
        net.setInput(cv2.dnn.blobFromImage(L_resized))
        ab = net.forward()[0].transpose(1, 2, 0)
    except Exception as e:
        print(f"Error during model forward: {e}")
        return False

    ab = cv2.resize(ab, (w, h))
    out_lab = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    out_bgr = cv2.cvtColor(out_lab, cv2.COLOR_Lab2BGR)
    out_bgr = np.clip(out_bgr * 255, 0, 255).astype("uint8")
    out_bgr = adjust_brightness_contrast(out_bgr, brightness=-10, contrast=15)
    cv2.imwrite(output_path, out_bgr)
    return True

@app.route('/', methods=['GET', 'POST'])
def index():
    original_url = enhanced_url = None
    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename:
            filename = secure_filename(file.filename)
            orig_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(orig_path)

            output_name = f"output_{filename}"
            output_path = os.path.join(OUTPUT_FOLDER, output_name)

            print(f"Saved input image at: {orig_path}")
            print(f"Does file exist? {os.path.exists(orig_path)}")
            print(f"File size: {os.path.getsize(orig_path) if os.path.exists(orig_path) else 'N/A'} bytes")

            if not colorize_image_local(orig_path, output_path):
                return "Colorization failed", 400

            original_url = url_for('static', filename=f'uploads/{filename}')
            enhanced_url = url_for('static', filename=f'outputs/{output_name}')
    return render_template('index.html', original_url=original_url, enhanced_url=enhanced_url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
