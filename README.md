# YOLO Image Classifier

A Streamlit app that uses Ultralytics YOLO26 to classify a sample photo or an uploaded JPEG or PNG. It shows the most likely class and up to five predictions with confidence scores.

## Run locally

```bash
git clone https://github.com/mahdi-hajizadh/YOLO-Image-Classification.git
cd YOLO-Image-Classification
python -m venv .venv
```

Activate the environment with `.\.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` on Linux or macOS.

```bash
python -m pip install -r requirements.txt
python -m streamlit run yolo_app.py
```

Open the local page shown by Streamlit, optionally upload an image, then select **Classify Image**. The first prediction may download `yolo26n-cls.pt` if the weights are unavailable locally.
