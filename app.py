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

os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

MAX_DIMENSION = 2048

def resize_img(img):
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(h, w)
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def enhance_image(input_path, output_path):
    print(f"Reading image for enhancement: {input_path}")
    img = cv2.imread(input_path)
    if img is None:
        print("ERROR: Could not read input image (file may be corrupt or unreadable by OpenCV).")
        return False
    else:
        print(f"Image loaded successfully, shape: {img.shape}")

    # Denoise
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # Sharpen
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    # Upscale to 4K resolution (3840 x 2160) while preserving aspect ratio
    target_w = 3840
    target_h = 2160
    h, w = sharpened.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    upscaled = cv2.resize(sharpened, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Optional: place the upscaled image on a black 4K canvas if you want exact 4K output
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = upscaled

    cv2.imwrite(output_path, canvas)
    print(f"Enhanced 4K image saved to {output_path}")
    return True


@app.route('/', methods=['GET', 'POST'])
def index():
    print(f"UPLOAD_FOLDER: {UPLOAD_FOLDER}, exists? {os.path.exists(UPLOAD_FOLDER)}")
    print(f"OUTPUT_FOLDER: {OUTPUT_FOLDER}, exists? {os.path.exists(OUTPUT_FOLDER)}")
    print(f"cv2 version: {cv2.__version__}")

    original_url = enhanced_url = None
    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename:
            filename = secure_filename(file.filename)
            orig_path = os.path.join(UPLOAD_FOLDER, filename)
            print(f"Saving file to {orig_path}")
            try:
                file.save(orig_path)
                print(f"File saved successfully.")
            except Exception as e:
                print(f"ERROR: Failed to save file: {e}")
                return "File save failed", 400

            if os.path.exists(orig_path):
                print(f"Saved file size: {os.path.getsize(orig_path)} bytes")
            else:
                print("ERROR: File does not exist after saving!")
                return "File save failed", 400

            output_name = f"enhanced_{filename}"
            output_path = os.path.join(OUTPUT_FOLDER, output_name)

            if not enhance_image(orig_path, output_path):
                print("Enhancement failed inside enhance_image")
                return "Enhancement failed", 400

            original_url = url_for('static', filename=f'uploads/{filename}')
            enhanced_url = url_for('static', filename=f'outputs/{output_name}')
        else:
            print("No file uploaded or empty filename.")
            return "No file uploaded", 400

    return render_template('index.html', original_url=original_url, enhanced_url=enhanced_url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
