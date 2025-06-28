import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
import traceback

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

# --- Colorization Model Load ---
COLORIZATION_MODEL_AVAILABLE = False
try:
    model_folder = os.path.join(BASE_DIR, 'models')
    proto_path = os.path.join(model_folder, 'colorization_deploy_v2.prototxt')
    model_path = os.path.join(model_folder, 'colorization_release_v2.caffemodel')
    pts_path = os.path.join(model_folder, 'pts_in_hull.npy')

    print("Loading colorization model:")
    print(f"  Proto: {proto_path}")
    print(f"  Model: {model_path}")
    print(f"  Pts:   {pts_path}")

    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    pts_in_hull = np.load(pts_path)

    class8 = net.getLayerId('class8_ab')
    conv8 = net.getLayerId('conv8_313_rh')
    pts = pts_in_hull.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype(np.float32)]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype=np.float32)]

    COLORIZATION_MODEL_AVAILABLE = True
    print("✅ Colorization model loaded successfully.")
except Exception:
    print("❌ Failed to load colorization model. Colorization disabled.")
    traceback.print_exc()

# --- Constants ---
MAX_DIMENSION = 7680  # 8K max dimension (width or height)

# --- Utility Functions ---
def boost_saturation(img_bgr, factor=1.1):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def adjust_brightness_contrast(img, brightness=0, contrast=20):
    alpha = 1 + (contrast / 100.0)
    beta = brightness
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def is_grayscale(img):
    if len(img.shape) == 2:
        return True
    if len(img.shape) == 3 and img.shape[2] == 1:
        return True
    if len(img.shape) == 3 and img.shape[2] == 3:
        b, g, r = cv2.split(img)
        return np.allclose(b, g, atol=10) and np.allclose(b, r, atol=10)
    return False

def resize_max_8k(img):
    h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim > MAX_DIMENSION:
        scale = MAX_DIMENSION / max_dim
        new_w, new_h = int(w * scale), int(h * scale)
        print(f"Resizing from ({w}, {h}) to ({new_w}, {new_h}) to fit max 8K")
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

# --- Image Inpainting ---
def inpaint_image_local(input_path, mask_path, output_path):
    img = cv2.imread(input_path)
    if img is None:
        print(f"❌ Error: Failed to load image from {input_path}")
        return False

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"❌ Error: Failed to load mask from {mask_path}")
        return False

    # Resize mask if needed
    if img.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    try:
        # Inpaint damaged areas
        inpainted = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        # Denoise and sharpen
        denoised = cv2.fastNlMeansDenoisingColored(inpainted, None, 3, 3, 7, 21)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        enhanced = adjust_brightness_contrast(sharpened, brightness=-10, contrast=15)

        Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)).save(output_path)
        print(f"✅ Inpainted image saved to: {output_path}")
        return True
    except Exception:
        print("❌ Inpainting failed.")
        traceback.print_exc()
        return False

# --- Main Processing ---
def colorize_image_local(input_path, output_path):
    img = cv2.imread(input_path)
    if img is None:
        print(f"❌ Error: Failed to load image from {input_path}")
        return False

    img = resize_max_8k(img)

    if not COLORIZATION_MODEL_AVAILABLE:
        print("⚠️ Colorization model unavailable.")
        return False

    try:
        scaled = img.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50  # Centering

        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0].transpose((1, 2, 0))
        ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

        L_orig = cv2.split(lab)[0]
        lab_out = np.concatenate((L_orig[:, :, np.newaxis], ab), axis=2)
        colorized_bgr = cv2.cvtColor(lab_out, cv2.COLOR_Lab2BGR)
        colorized_bgr = np.clip(colorized_bgr, 0, 1)
        colorized_bgr = (colorized_bgr * 255).astype("uint8")
    except Exception:
        print("❌ Colorization failed.")
        traceback.print_exc()
        return False

    try:
        denoised = cv2.fastNlMeansDenoisingColored(colorized_bgr, None, 3, 3, 7, 21)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        enhanced = adjust_brightness_contrast(sharpened, brightness=-10, contrast=15)

        Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)).save(output_path)
        print(f"✅ Enhanced image saved to: {output_path}")
        return True
    except Exception:
        print("❌ Enhancement failed.")
        traceback.print_exc()
        return False

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    original_url = None
    enhanced_url = None

    if request.method == 'POST':
        file = request.files.get('image')
        mask_file = request.files.get('mask')  # Optional mask for inpainting

        if file and file.filename:
            filename = secure_filename(file.filename)
            name, ext = os.path.splitext(filename)

            # Save uploaded original image
            orig_desktop_path = os.path.join(MY_IMAGE_FOLDER, f"{name}_original{ext}")
            file.save(orig_desktop_path)

            file.seek(0)
            orig_static_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(orig_static_path)

            # Paths for intermediate and final images
            inpainted_desktop_path = os.path.join(MY_IMAGE_FOLDER, f"{name}_inpainted{ext}")
            inpainted_static_name = f"inpainted_{name}.png"
            inpainted_static_path = os.path.join(OUTPUT_FOLDER, inpainted_static_name)

            # If mask provided, do inpainting first
            if mask_file and mask_file.filename:
                mask_filename = secure_filename(mask_file.filename)
                mask_desktop_path = os.path.join(MY_IMAGE_FOLDER, mask_filename)
                mask_file.save(mask_desktop_path)

                success = inpaint_image_local(orig_desktop_path, mask_desktop_path, inpainted_desktop_path)
                if not success:
                    return "❌ Failed to inpaint image.", 400
                # Copy inpainted image to static folder
                Image.open(inpainted_desktop_path).save(inpainted_static_path)

                # Colorize using inpainted image
                success = colorize_image_local(inpainted_desktop_path, inpainted_desktop_path)
                if not success:
                    return "❌ Failed to colorize inpainted image.", 400

                enhanced_url = url_for('static', filename=f'outputs/{inpainted_static_name}')
                original_url = url_for('static', filename=f'uploads/{filename}')
            else:
                # No inpainting mask, just colorize original
                enhanced_static_name = f"enhanced_{name}.png"
                enhanced_static_path = os.path.join(OUTPUT_FOLDER, enhanced_static_name)
                success = colorize_image_local(orig_desktop_path, enhanced_static_path)
                if not success:
                    return "❌ Failed to colorize image.", 400

                enhanced_url = url_for('static', filename=f'outputs/{enhanced_static_name}')
                original_url = url_for('static', filename=f'uploads/{filename}')

    return render_template('index.html', original_url=original_url, enhanced_url=enhanced_url)

if __name__ == '__main__':
    app.run(debug=True)
