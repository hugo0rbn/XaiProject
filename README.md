# Deepfake Audio Detector with XAI

## Files not in repo
- chexpert_convnextv2_tiny_384
- vit_chest_xray
These models can be downloaded using the notebook HF_model_downloader

## Quick Start

```bash
conda activate base
conda env remove -n XAI_env -y
conda create -n XAI_env python=3.8 -y
conda activate XAI_env
pip install -r requirements.txt

streamlit run app.py
```

## Project Overview
This project provides a unified Streamlit interface for:

- **Deepfake audio detection** from `.wav` files (audio converted to MEL spectrograms and classified with a CNN model).
- **Chest X-Ray lung pathology detection** from images (`.png/.jpg/.jpeg`) using offline Hugging Face vision models.

In both modalities, the application integrates **Explainable AI (XAI)** methods (LIME, SHAP, and Grad-CAM / Attention Rollout when applicable) to help interpret model outputs.

## Group Information
- **TD group number**: DIA 5
- **Group members**:
  - Hugo ROBIN 
  - Ghadi SALAMEH
  - Hector MELL MARIOLLE
  - Jules SAYAD-BARTH

## Technologies Used
- **Deep Learning Models**:
  - **Audio**: CNN classifier in **TensorFlow/Keras** (trained on MEL spectrogram images; loaded from `Streamlit/saved_model/model/`)
  - **X-Ray**: **PyTorch** image classification models from **offline Hugging Face** snapshots (ConvNeXtV2 and ViT)
- **X-Ray Models (offline Hugging Face)**:
  - ConvNeXtV2 CheXpert-5 (local snapshot directory)
  - ViT Chest X-Ray (local snapshot directory)
- **Explainable AI (XAI) Techniques**: LIME, Grad-CAM, SHAP (plus Attention Rollout for ViT-based models)
- **Data Processing**: Spectrogram conversion for audio data
- **Programming Languages and Libraries**: Python, TensorFlow, Keras, Matplotlib, NumPy
- **Development Tools**: Jupyter Notebooks
- **Web Application Framework**: Streamlit for interactive web app deployment

## Key Features
- **Audio**: TensorFlow/Keras CNN classifier on MEL spectrograms for real/fake audio detection.
- **X-Ray**: Two selectable models (ConvNeXtV2 and ViT) for lung pathology detection with 5 output labels each.
- **XAI Methods**: LIME, SHAP, Grad-CAM (for CNN models), Attention Rollout (for ViT models).
- **Automatic compatibility filtering**: Grad-CAM is hidden for ViT; Attention Rollout is hidden for CNN models.
- **Unified interface**: one upload zone, modality auto-detection (audio vs X-Ray), and filtered XAI options.
- **Comparison view**: side-by-side comparison of selected XAI outputs.

## Dataset
The dataset used for training our deepfake audio detection models is the 'Fake or Real' dataset, created by researchers from York University. This dataset consists of authentic and deepfake audio recordings that have been used to train our models to distinguish between real and fake samples effectively.

For enhanced model performance, the audio files were converted into spectrograms. Spectrograms are visual representations of the spectrum of frequencies in a sound or other signal as they vary with time, which provides a more informative feature set for deep learning models.

![Example Spectrogram](https://raw.githubusercontent.com/Aamir-Hullur/Deepfake-Audio-detection-using-XAI/main/img/spectrogram_example.png)

## Model Performance
![Model Performance](https://raw.githubusercontent.com/Aamir-Hullur/Deepfake-Audio-detection-using-XAI/main/img/Model_performanc.png)

## XAI Model Comparison
![XAI Model Comparison](https://raw.githubusercontent.com/Aamir-Hullur/Deepfake-Audio-detection-using-XAI/main/img/XAI_model_comparison.png)

## Streamlit Web Application

This project includes a Streamlit web application that provides a user-friendly interface for interacting with the deepfake audio detection models. Below is a preview of the application in action.

![Streamlit App Demo](https://raw.githubusercontent.com/Aamir-Hullur/Deepfake-Audio-detection-using-XAI/main/img/Streamlit_demo.gif)

## Setup and Installation

### Prerequisites
- Python **3.8** (recommended to match TensorFlow 2.6.x compatibility)
- A working `pip` environment (venv/conda)

### Install dependencies
From the repository root:

```bash
pip install -r Streamlit/requirements.txt
```

### Local model files (required)
This project uses **offline/local models** for the X-Ray part. Ensure these directories exist:

- `Streamlit/saved_model/chexpert_convnextv2_tiny_384/` (hugging face : shreydan/CheXpert-5-convnextv2-tiny-384)
- `Streamlit/saved_model/vit_chest_xray/` (hugging face : codewithdark/vit-chest-xray)

Audio model directory:
- `Streamlit/saved_model/model/`

If your paths differ, update them in `Streamlit/app.py`.

## Run the Interface
From the repository root:

```bash
cd Streamlit
streamlit run app.py
```

Then open the URL printed by Streamlit (typically `http://localhost:8501`).

## Demo Instructions (Teacher/TD)

### Recommended demo path: “Unified”
1. In the sidebar, select **Unified**.
2. Upload one of:
   - an **audio** file (`.wav`) → the app auto-runs the audio model and shows predictions + XAI.
   - an **X-Ray image** (`.png/.jpg/.jpeg`) → the app auto-runs the selected X-Ray model and shows predictions + XAI.
3. Use **Explain / Compare** to:
   - pick XAI methods (incompatible methods are automatically hidden)
   - compare selected XAI outputs side-by-side

### What to say when presenting
- **LIME**: Local explanation by perturbing the input; highlighted regions (bounded areas) show superpixels that most influenced the prediction. Note: on spectrograms, superpixels do not map cleanly to time-frequency structure.
- **SHAP**: Feature/region contribution values based on a game-theoretic approach. Red regions support the prediction, blue regions oppose it.
- **Grad-CAM**: Convolutional-model heatmap based on gradients. Brighter/red areas indicate stronger influence on the prediction (only for CNN models).
- **Attention Rollout**: Transformer-specific attention aggregation (used for ViT models instead of Grad-CAM). Shows which image regions the model attended to.

## Notes / Limitations
- X-Ray outputs are **not** medical advice; they are for education/research only.
- XAI methods can be computationally expensive, especially SHAP.
- **LIME on spectrograms**: LIME superpixels do not map cleanly to time-frequency structure; results should be interpreted as approximate regions of interest rather than precise frequency bands.
- **Multi-label X-Ray**: The current XAI visualizations explain the top predicted pathology; for clinical use, each pathology should be explained separately.

## References
- **Audio Detection**: Based on [Deepfake-Audio-Detection-with-XAI](https://github.com/Guri10/Deepfake-Audio-Detection-with-XAI) by Aamir Hullur, Atharva Gurav, Aditi Govindu, Parth Godse.
- **X-Ray Models**: [shreydan/CheXpert-5-convnextv2-tiny-384](https://huggingface.co/shreydan/CheXpert-5-convnextv2-tiny-384) and [codewithdark/vit-chest-xray](https://huggingface.co/codewithdark/vit-chest-xray) from Hugging Face.
