import os
import cv2
import numpy as np
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename

# Import Real-ESRGAN
from realesrgan import RealESRGAN

app = Flask(__name__, static_folder='static')

# --- Directory Setup ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, 'uploads')
OUTPUT_FOLDER = os.path.join(STATIC_FOLDER, 'outputs')

os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize Real-ESRGAN model once when app starts
device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
print(f"Using device: {device}")

sr_model = RealESRGAN(device, scale=4)  # 4x upscaling
sr_model.load_weights('RealESRGAN_x4.pth')  # Make sure the weights file is available in your environment

def enhance_image_realesrgan(input_path, output_path):
    print(f"Reading image for Real-ESRGAN enhancement: {input_path}")
    img = cv2.imread(input_path)
    if img is None:
        print("ERROR: Could not read input image.")
        return False

    # Convert BGR to RGB for Real-ESRGAN
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run Real-ESRGAN super resolution
    sr_img = sr_model.predict(img_rgb)

    # Convert back to BGR for saving with OpenCV
    sr_img_bgr = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, sr_img_bgr)
    print(f"Real-ESRGAN enhanced image saved to {output_path}")
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

            output_name = f"enhanced_{filename}"
            output_path = os.path.join(OUTPUT_FOLDER, output_name)

            if not enhance_image_realesrgan(orig_path, output_path):
                return "Enhancement failed", 400

            original_url = url_for('static', filename=f'uploads/{filename}')
            enhanced_url = url_for('static', filename=f'outputs/{output_name}')
        else:
            return "No file uploaded", 400

    return render_template('index.html', original_url=original_url, enhanced_url=enhanced_url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
