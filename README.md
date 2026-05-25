# ThyroVision

ThyroVision is an image-analysis application for binary classification of thyroid medical images as benign or malignant. It uses a TensorFlow/PyTorch implementation of `FibonacciNet`, presents a Grad-CAM attention overlay with each result, and can generate a DOCX analysis report.

This project is intended for research, demonstration, and decision-support workflows. It is not a medical device and must not be used as a substitute for review by a qualified clinician.

## What the application provides

- Image upload for PNG and JPEG files.
- Binary prediction: `Benign (Non-Cancerous)` or `Malignant (Cancerous)`.
- Confidence information returned with each prediction.
- Grad-CAM overlay showing regions that influenced the network output.
- Downloadable DOCX report containing the image, result, confidence, and heatmap when available.
- REST endpoints for applications that need to integrate inference or report generation.

## How inference works

An uploaded image is converted to RGB, resized to `224 x 224`, and scaled to values between `0` and `1`. The model returns a single score; scores greater than `0.5` are reported as malignant.

The application loads `thyroid_fibonaccinet_best.keras` from the [ThyroVision model repository on Hugging Face](https://huggingface.co/SimonPathula/thyrovision) when it starts. An internet connection is therefore required on the first run, or whenever the model is not already present in the local Hugging Face cache.

## Run the local application

The Streamlit interface is the recommended way to run ThyroVision locally. It performs inference and report generation within the local Python process after the model has been downloaded.

### Requirements

- Python 3.11 or later
- An internet connection for the initial model download

### Installation

From the project directory, create a virtual environment and install the required packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

### Start Streamlit

```powershell
python -m streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit, then:

1. Upload a `.png`, `.jpg`, or `.jpeg` image.
2. Review the predicted class and confidence information.
3. Review the Grad-CAM overlay, if generated successfully.
4. Download the DOCX report from the analysis page.

## Run the API service

The FastAPI service is available for programmatic access:

```powershell
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

On startup, the service downloads and loads the same model used by the Streamlit application. Once the service is ready, it exposes these endpoints:

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/` | `GET` | Serves the bundled browser interface |
| `/analyze` | `POST` | Accepts an uploaded image and returns prediction data and encoded images |
| `/report` | `POST` | Accepts an uploaded image and returns a generated DOCX report |

Example requests:

```powershell
curl.exe -X POST -F "file=@tests/0000.jpg" http://localhost:8000/analyze
curl.exe -X POST -F "file=@tests/0000.jpg" -o thyroid_analysis_report.docx http://localhost:8000/report
```

The browser interface stored in `frontend/` is currently configured to submit images to the deployed API at `thyrovisionxai.onrender.com`, even when the page is served from a local FastAPI process. Use the Streamlit application for local-only analysis, or change the request URLs in `frontend/static/app.js` when configuring a self-hosted browser frontend.

## Project structure

| Path | Description |
| --- | --- |
| `streamlit_app.py` | Local interactive application for upload, analysis, Grad-CAM, and report download |
| `app.py` | FastAPI application setup and service startup |
| `backend/routes.py` | Analysis and report endpoints |
| `utils/` | Image preprocessing, model definition, Grad-CAM, reporting, configuration, and logging |
| `frontend/` | HTML, JavaScript, and CSS served by the FastAPI application |
| `vercel-frontend/` | Static frontend prepared for separate hosting |
| `models/` | Locally stored trained model artifacts and training history |
| `src/` | Model and training scripts |
| `tests/` | Sample images available for manual inference checks |
| `notebooks/` | Experimental training, evaluation, and interpretability notebooks |

## Model training

Training is not required to run the application; inference uses the model hosted on Hugging Face.

For retraining, `src/train_v2.py` implements the model architecture used by the inference utilities and writes resulting model files to `models/`. Before running it, update the script's `BASE_PATH` value to the location of a dataset arranged as:

```text
Thyroid Data/
|-- 0/
|   |-- image files for the benign class
|-- 1/
    |-- image files for the malignant class
```

The training script applies image resizing and normalization, balances classes by oversampling, creates train/validation/test splits, and trains a binary classifier with checkpointing and early stopping.

## Responsible use

ThyroVision produces an automated classification and an explanatory heatmap; neither is a diagnosis. Model outputs may be incorrect, incomplete, or sensitive to acquisition quality and dataset differences. Do not use results for patient-care decisions without appropriate clinical review and validation for the intended setting.

Medical images may contain sensitive information. Confirm where images are processed before uploading clinical material: the Streamlit workflow runs analysis in the local process after obtaining the model, while the bundled browser frontend is presently configured to send uploads to a hosted API.
