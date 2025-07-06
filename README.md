# 🧠 Image De-Photobombing using Deep Learning

Image De-Photobombing is a cutting-edge deep learning project designed to automatically detect, segment, and remove photobombers or unwanted objects from photographs using intelligent inpainting and masking techniques.

> This solution utilizes advanced segmentation models to generate object masks and uses deep learning-based inpainting (LaMa or equivalent) to realistically fill the removed regions.

---

## 🚀 Features

- 🎯 Automatic detection and removal of unwanted objects
- 🖼️ Smart segmentation using deep learning (e.g., SAM or custom mask logic)
- 🧽 Realistic inpainting for photobombed regions using LaMa or similar
- 🌐 Web-based interface for easy usage
- 📦 Modular code: `app.py`, `mask.py`, `run_inpaint.py`

---

## 🛠️ Installation
```bash
Create Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

python app.py
