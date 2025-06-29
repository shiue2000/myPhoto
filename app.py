import os
import cv2
import numpy as np
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static')

# --- Directory Setup ---
DESKTOP_DIR = os.path.join(os.path.expanduser("~"), "Desktop")
MY_IMAGE_FOLDER = os.path.join(DESKTOP_DIR, "myImage")
os.makedirs(MY_IMAGE_FOLDER, exist_ok=True)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, 'uploads')
OUTPUT_FOLDER = os.path.join(STATIC_FOLDER, 'outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Colorization Model ---
COLORIZATION_MODEL_AVAILABLE = False
try:
    net = cv2.dnn.readNetFromCaffe(
        os.path.join(BASE_DIR, 'colorization_deploy_v2.prototxt'),
        os.path.join(BASE_DIR, 'colorization_release_v2.caffemodel')
    )
    pts_in_hull = np.load(os.path.join(BASE_DIR, 'pts_in_hull.npy'))
    pts = pts_in_hull.transpose().reshape(2, 313, 1, 1).astype(np.float32)
    net.getLayer(net.getLayerId('class8_ab')).blobs = [pts]
    net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, dtype=np.float32)]
    COLORIZATION_MODEL_AVAILABLE = True
except Exception:
    pass  # Model load failed; colorization disabled

MAX_DIMENSION = 2048  # 💡 Reduce max resolution to save memory

# --- Utilities ---
def resize_img(img):
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(h, w)
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def adjust_brightness_contrast(img, brightness=0, contrast=20):
    alpha = 1 + contrast / 100.0
    return cv2.convertScaleAbs(img, alpha=alpha, beta=brightness)

def inpaint_image_local(input_path, mask_path, output_path):
    img = cv2.imread(input_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        return False
    img = resize_img(img)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    result = adjust_brightness_contrast(result, brightness=-10, contrast=15)
    cv2.imwrite(output_path, result)
    del img, mask, result
    return True

def colorize_image_local(input_path, output_path):
    if not COLORIZATION_MODEL_AVAILABLE:
        return False
    img = cv2.imread(input_path)
    if img is None:
        return False
    img = resize_img(img)
    lab = cv2.cvtColor(img.astype("float32") / 255.0, cv2.COLOR_BGR2LAB)
    L = cv2.resize(lab[:, :, 0], (224, 224)) - 50
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0].transpose(1, 2, 0)
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
    out_lab = np.concatenate((lab[:, :, 0:1], ab), axis=2)
    out_bgr = cv2.cvtColor(out_lab, cv2.COLOR_Lab2BGR)
    out_bgr = np.clip(out_bgr * 255, 0, 255).astype("uint8")
    out_bgr = adjust_brightness_contrast(out_bgr, brightness=-10, contrast=15)
    cv2.imwrite(output_path, out_bgr)
    del img, lab, L, ab, out_lab, out_bgr
    return True

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    original_url = enhanced_url = None
    if request.method == 'POST':
        file = request.files.get('image')
        mask_file = request.files.get('mask')
        if file and file.filename:
            filename = secure_filename(file.filename)
            orig_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(orig_path)

            output_name = f"output_{filename}"
            output_path = os.path.join(OUTPUT_FOLDER, output_name)

            if mask_file and mask_file.filename:
                mask_path = os.path.join(UPLOAD_FOLDER, secure_filename(mask_file.filename))
                mask_file.save(mask_path)
                if not inpaint_image_local(orig_path, mask_path, output_path):
                    return "Inpainting failed", 400
            else:
                if not colorize_image_local(orig_path, output_path):
                    return "Colorization failed", 400

            original_url = url_for('static', filename=f'uploads/{filename}')
            enhanced_url = url_for('static', filename=f'outputs/{output_name}')
    return render_template('index.html', original_url=original_url, enhanced_url=enhanced_url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
