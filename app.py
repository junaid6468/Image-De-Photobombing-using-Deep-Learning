from flask import Flask, render_template, request, redirect
import os
from werkzeug.utils import secure_filename
from model_loader import load_model, process_image
import cv2
from datetime import datetime
import shutil
import torch
import numpy as np
from PIL import Image   
from run_inpaint import inpaint_image
import time
# from util import move_images

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'dataset'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = load_model('de_photobomb_22.05psnr_2025-04-16_18-39-17.pt', device)

def move_images():
    # Define paths
    original_folder = 'dataset'
    output_folder = 'output_images'

    # Static destination folders
    static_dir = 'static'

    # Create static folders if they don't exist
    os.makedirs(static_dir, exist_ok=True)

    # Move images from original_folder to static/original
    for filename in os.listdir(original_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')) and not filename.lower().endswith(('_mask.png', '_mask.jpg', '_mask.jpeg', '_mask.bmp', '_mask.tiff')):
            src_path = os.path.join(original_folder, filename)
            dst_path = os.path.join(static_dir, filename)
            shutil.copy2(src_path, dst_path)  # use copy2 to preserve metadata

    # Move images from output_folder to static/output
    for filename in os.listdir(output_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            src_path = os.path.join(output_folder, filename)
            dst_path = os.path.join(static_dir, filename)
            shutil.copy2(src_path, dst_path)

    print("Images copied to static folder.")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}")
            mask_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename.split('.')[0]}_mask.jpg")
            
            # Save input
            file.save(input_path)
            
            # Process image
            maskStart_time = time.time()
            mask_img = process_image(input_path)
            maskEnd_time = time.time()
            print("Mask Creation Time: ", maskEnd_time - maskStart_time)
            cv2.imwrite(mask_path, mask_img)
            inpaintStart_time = time.time()
            res = inpaint_image()
            inpaintEnd_time = time.time()
            print("Inpainting Time: ", inpaintEnd_time - inpaintStart_time)
            if res:
                move_images()
                return render_template('index.html', 
                                    input_img=f"{filename}",
                                    output_img=f"{filename.split('.')[0]}_mask.png",)
            else:
                print("Inpainting failed.")

    return render_template('index.html')


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0', port=8000, debug=True)