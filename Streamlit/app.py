import os
import hashlib

import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
import tensorflow as tf
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor

st.set_page_config(page_title="XAI Detection System", page_icon="")

# Audio class names
audio_class_names = ["real", "fake"]

# X-Ray: only two offline Hugging Face snapshot directories (local)
XRAY_VIT_DIR = os.path.join("saved_model", "vit_chest_xray")
XRAY_CONVNEXT_DIR = os.path.join("saved_model", "chexpert_convnextv2_tiny_384")

# X-Ray labels (provided by you)
VIT_CHEST_XRAY_LABELS = ["Cardiomegaly", "Edema", "Consolidation", "Pneumonia", "No Finding"]
CONVNEXT_CHEXPERT5_LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]

# =========================
# XAI INTERPRETATION NOTES
# =========================

def xai_note(method: str, modality: str, model_key: str = "") -> str:
    """Short interpretation note shown next to each XAI method."""
    method = (method or "").strip().lower()
    modality = (modality or "").strip().lower()

    if method == "lime":
        if modality == "audio":
            return (
                "LIME perturbs parts of the spectrogram and fits a local surrogate model. "
                "Highlighted regions (bounded areas) are the superpixels that most influenced the prediction. "
                "Note: LIME superpixels on spectrograms do not map cleanly to time-frequency structure; "
                "interpret as approximate regions of interest rather than precise frequency bands."
            )
        return (
            "LIME perturbs parts of the input and fits a simple local surrogate model. "
            "Highlighted regions show superpixels that contributed to the prediction. "
            "LIME is local (one example) and can vary slightly between runs."
        )
    if method == "shap":
        return (
            "SHAP assigns an importance value to each region of the input based on a game-theoretic approach. "
            "In the heatmap, red typically supports the prediction and blue opposes it. "
            "SHAP is more computationally expensive than LIME."
        )
    if method in ("grad-cam", "gradcam"):
        if modality == "audio":
            return (
                "Grad-CAM highlights spectrogram regions that most influenced the model output by backpropagating "
                "gradients to the last convolutional layer. Brighter/red areas indicate stronger influence on the "
                "real/fake classification decision."
            )
        return (
            "Grad-CAM highlights the regions that most influenced the model output by backpropagating gradients "
            "to the last convolutional feature maps. Brighter/red areas indicate stronger influence. "
            "Grad-CAM is only applicable to convolutional models."
        )
    if method in ("attention rollout", "attention-rollout", "attention"):
        return (
            "Attention rollout is a transformer-specific explanation that aggregates attention across layers "
            "to estimate which image regions influenced the decision. It is not a true gradient-based CAM."
        )
    return ""


def show_xai_note(method: str, modality: str, model_key: str = ""):
    note = xai_note(method, modality, model_key)
    if note:
        st.info(note)

# ============== AUDIO FUNCTIONS (DEEPFAKE DETECTION) ==============

def save_audio_bytes(filename: str, audio_bytes: bytes) -> str:
    """Save uploaded audio bytes and return the local path."""
    os.makedirs("audio_files", exist_ok=True)
    path = os.path.join("audio_files", filename)
    with open(path, "wb") as f:
        f.write(audio_bytes)
    return path


def _sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _xai_cache_get():
    return st.session_state.setdefault("xai_cache", {})


def _xai_cached(key: str, compute_fn):
    cache = _xai_cache_get()
    if key in cache:
        return cache[key]
    val = compute_fn()
    cache[key] = val
    return val


def create_spectrogram(sound):
    """Create a deterministic 224x224 RGB MEL spectrogram image.

    Important: keep this generation consistent with the training pipeline.
    We therefore use librosa.display.specshow (as before), but force a fixed DPI/figsize
    so the output is stable and does not depend on the backend.
    """
    audio_file = sound if os.path.exists(sound) else os.path.join("audio_files", sound)

    # Match librosa default behavior used in the original pipeline (resample to 22050)
    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)

    # Fixed-size render: 224x224 pixels
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    # Matplotlib default colormap is "viridis" (keep consistent)
    librosa.display.specshow(log_ms, sr=sr, cmap="viridis")

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)

    # Safety: enforce exact input size
    buf_resized = cv2.resize(buf, (224, 224), interpolation=cv2.INTER_AREA)
    st.image(buf_resized)
    return buf_resized


def predictions_audio(image_data, model):
    """Audio model prediction"""
    img_array = np.array(image_data)
    img_array1 = img_array / 255
    img_batch = np.expand_dims(img_array1, axis=0)
    prediction = model.predict(img_batch)
    class_label = np.argmax(prediction)
    return class_label, prediction


def lime_predict_audio(image_data, model):
    """LIME explainability for audio model"""
    img_array = np.array(image_data)
    img_array1 = img_array / 255
    img_batch = np.expand_dims(img_array1, axis=0)

    def predict_fn(x):
        x = np.array(x).astype(np.float32)
        return model.predict(x)

    prediction = predict_fn(img_batch)
    class_label = int(np.argmax(prediction))

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_array1.astype('float64'),
        predict_fn,
        hide_color=0,
        num_samples=1000
    )
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    temp, mask = explanation.get_image_and_mask(
        int(np.argmax(prediction[0], axis=0)),
        positive_only=False,
        num_features=8,
        hide_rest=True
    )
    axs[0].imshow(image_data)
    axs[0].set_title("Original Spectrogram")
    axs[0].axis("off")
    axs[1].imshow(mark_boundaries(temp, mask))
    axs[1].set_title(f"LIME Explanation - {audio_class_names[class_label]}")
    axs[1].axis("off")
    plt.tight_layout()
    return fig


def grad_predict_audio(image_data, model_mob, class_idx):
    """Grad-CAM for the *actual* audio classifier model (not ImageNet VGG16).
    
    Args:
        image_data: PIL Image or array of the spectrogram
        model_mob: The TensorFlow/Keras audio classification model
        class_idx: Index of the class to explain (0=real, 1=fake)
    """
    # Prepare input
    img_array = np.array(image_data).astype(np.float32) / 255.0
    x = np.expand_dims(img_array, axis=0)

    # Find last Conv2D layer in the provided model
    last_conv_layer = None
    for layer in reversed(model_mob.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
    if last_conv_layer is None:
        st.warning("Grad-CAM is not available for this audio model (no Conv2D layer found).")
        return None

    grad_model = tf.keras.models.Model([model_mob.inputs], [last_conv_layer.output, model_mob.output])

    with tf.GradientTape() as tape:
        conv_out, preds_out = grad_model(x)
        if preds_out.shape[-1] == 1:
            # Binary sigmoid: use that neuron
            class_channel = preds_out[:, 0]
        else:
            class_channel = preds_out[:, class_idx]
    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Resize heatmap to input size (224x224)
    heatmap = cv2.resize(heatmap.astype(np.float32), (x.shape[2], x.shape[1]))

    # Overlay
    img_vis = (x[0] * 255.0).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_vis, 0.6, heatmap_color, 0.4, 0)

    fig1, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img_vis)
    ax[0].set_title("Input")
    ax[0].axis("off")
    ax[1].imshow(heatmap, cmap="jet")
    ax[1].set_title("Grad-CAM Heatmap")
    ax[1].axis("off")
    ax[2].imshow(overlay)
    ax[2].set_title(f"Overlay - {audio_class_names[class_idx]}")
    ax[2].axis("off")
    plt.tight_layout()
    return fig1


def shap_predict_audio(image_data, model):
    """SHAP explainability for audio model (robust, model-agnostic image masker)."""
    img_array = np.array(image_data).astype(np.float32)
    img_array1 = img_array / 255.0

    # Ensure shape is (224,224,3)
    if img_array1.ndim == 2:
        img_array1 = np.stack([img_array1] * 3, axis=-1)
    if img_array1.shape[-1] == 4:
        img_array1 = img_array1[:, :, :3]

    def predict_fn(x):
        """x: (batch,224,224,3) in [0,1]. Returns (batch,2)."""
        x = np.array(x).astype(np.float32)
        preds = model.predict(x)

        # If model is binary sigmoid (batch,1), convert to 2-class probs
        if preds.ndim == 2 and preds.shape[1] == 1:
            p = preds[:, 0]
            preds = np.stack([1.0 - p, p], axis=1)
        return preds

    masker = shap.maskers.Image("blur(16,16)", img_array1.shape)
    explainer = shap.Explainer(predict_fn, masker, output_names=audio_class_names)

    shap_exp = explainer(img_array1[np.newaxis, :], max_evals=200, batch_size=10)

    pred = predict_fn(img_array1[np.newaxis, :])[0]
    class_label = int(np.argmax(pred))

    sv = shap_exp.values[0, :, :, :, class_label]
    shap_sum = np.sum(np.abs(sv), axis=-1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_array1)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(shap_sum, cmap='hot')
    axes[1].set_title(f"SHAP Values - {audio_class_names[class_label]}")
    axes[1].axis('off')

    axes[2].imshow(img_array1)
    axes[2].imshow(shap_sum, cmap='hot', alpha=0.5)
    axes[2].set_title("SHAP Overlay")
    axes[2].axis('off')

    plt.tight_layout()
    return fig

# ============== X-RAY FUNCTIONS (LUNG PATHOLOGY DETECTION) ==============

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def _numpy_compat_aliases_for_torch_load():
    """Compatibility shim for torch.load on pickles created with newer NumPy.

    Some checkpoints reference modules like `numpy._core` (newer NumPy internal layout),
    while older NumPy versions (e.g., 1.19.x pinned by TF 2.6) only expose `numpy.core`.
    We provide module aliases so unpickling can resolve them.
    """
    try:
        import sys
        import numpy as np

        sys.modules.setdefault("numpy._core", np.core)
        if hasattr(np.core, "multiarray"):
            sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
        if hasattr(np.core, "_multiarray_umath"):
            sys.modules.setdefault("numpy._core._multiarray_umath", np.core._multiarray_umath)
        if hasattr(np.core, "numeric"):
            sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    except Exception:
        pass


@st.experimental_singleton(suppress_st_warning=True)
def load_xray_model(model_key: str):
    """Load one of the two offline HF snapshot directories for X-Ray."""
    clear_gpu_memory()
    device = "cpu"

    if model_key == "chexpert_convnextv2_tiny_384":
        local_dir = XRAY_CONVNEXT_DIR
        labels = CONVNEXT_CHEXPERT5_LABELS
        default_size = 384
    elif model_key == "vit_chest_xray":
        local_dir = XRAY_VIT_DIR
        labels = VIT_CHEST_XRAY_LABELS
        default_size = 224
    else:
        raise ValueError(f"Unknown X-Ray model_key: {model_key}")

    if not os.path.isdir(local_dir):
        raise FileNotFoundError(f"Missing local HF model directory: {local_dir}")

    processor = AutoImageProcessor.from_pretrained(local_dir, local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(
        local_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
    ).to(device)
    model.eval()

    # Force the labels you provided for consistent UI
    model.config.id2label = {i: labels[i] for i in range(len(labels))}
    model.config.label2id = {labels[i]: i for i in range(len(labels))}

    # Determine target input size for UI preview
    size = getattr(processor, "size", None)
    if isinstance(size, dict):
        input_size = int(size.get("shortest_edge") or size.get("height") or size.get("width") or default_size)
    elif isinstance(size, int):
        input_size = int(size)
    else:
        input_size = int(default_size)

    return {
        "model": model,
        "processor": processor,
        "model_key": model_key,
        "labels": labels,
        "input_size": input_size,
    }


def preprocess_xray(image, model_bundle):
    """Preprocess X-Ray image for the two HF snapshot models."""
    processor = model_bundle["processor"]
    input_size = int(model_bundle.get("input_size", 224))

    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    # UI preview (grayscale, resized to the model input)
    img_gray = np.array(image.convert("L"))
    img_gray = cv2.resize(img_gray, (input_size, input_size))
    img_processed = img_gray[None, :, :]

    return inputs, img_processed


def _get_xray_labels(model_bundle):
    """Return output labels for the selected X-Ray model."""
    return model_bundle.get("labels", [])



def predictions_xray(image, model_bundle):
    """X-Ray model prediction (two HF snapshot models)."""
    model = model_bundle["model"]
    inputs, img_processed = preprocess_xray(image, model_bundle)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        preds = torch.sigmoid(logits).cpu().numpy()[0]

    labels = _get_xray_labels(model_bundle)
    results = list(zip(labels, preds.tolist()))
    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
    return results_sorted, inputs, img_processed


def grad_cam_xray(model, img_tensor, target_class_idx=None):
    """Grad-CAM for X-Ray model (PyTorch)"""
    model.eval()

    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    img_tensor.requires_grad_(True)

    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module

    if target_layer is None:
        return None

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    output = model(img_tensor)

    if target_class_idx is None:
        target_class_idx = output.argmax().item()

    model.zero_grad()
    output[0, target_class_idx].backward()

    forward_handle.remove()
    backward_handle.remove()

    grads = gradients[0]
    acts = activations[0]

    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * acts, dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()

    return cam


def grad_cam_hf_image_model(model, pixel_values: torch.Tensor, target_class_idx=None):
    """Grad-CAM for HF vision models with Conv layers (e.g., ConvNeXtV2)."""
    model.eval()

    device = next(model.parameters()).device
    pixel_values = pixel_values.to(device)

    target_layer = None
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
    if target_layer is None:
        return None

    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_inp, grad_out):
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    logits = model(pixel_values=pixel_values).logits
    scores = torch.sigmoid(logits)
    if target_class_idx is None:
        target_class_idx = int(scores[0].argmax().item())

    model.zero_grad()
    logits[0, target_class_idx].backward()

    h1.remove()
    h2.remove()

    acts = activations[0]
    grads = gradients[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam[0, 0].detach().cpu().numpy()
    return cam


def vit_attention_rollout_xray(model_bundle, image):
    """Attention rollout for ViT-based X-Ray models.

    This is the closest equivalent to a heatmap explanation for ViT models in this app,
    since classic Grad-CAM relies on convolutional feature maps.
    """
    model = model_bundle["model"]
    processor = model_bundle["processor"]
    device = next(model.parameters()).device

    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs, output_attentions=True)

    attentions = getattr(out, "attentions", None)
    if attentions is None:
        raise ValueError("Model did not return attentions (output_attentions=True not supported).")

    # attentions: tuple(num_layers) of (B, heads, tokens, tokens)
    attn_mat = None
    for layer_attn in attentions:
        # Mean over heads -> (tokens, tokens)
        a = layer_attn[0].mean(dim=0)
        # Add residual connection and normalize
        a = a + torch.eye(a.size(-1), device=a.device)
        a = a / a.sum(dim=-1, keepdim=True)
        attn_mat = a if attn_mat is None else attn_mat @ a

    # CLS token attention to the patches (exclude CLS itself)
    cls_attn = attn_mat[0, 1:]
    num_patches = cls_attn.numel()
    grid_size = int(np.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise ValueError(f"Unexpected number of patches: {num_patches}")

    heatmap = cls_attn.reshape(grid_size, grid_size).detach().cpu().numpy()
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-8)

    # Upsample to 224x224 for display
    heatmap_224 = cv2.resize(heatmap.astype(np.float32), (224, 224))
    return heatmap_224


def lime_predict_xray(image, model_bundle):
    """LIME explainability for X-Ray model (DenseNet or ViT)."""
    model = model_bundle["model"]
    model_key = model_bundle["model_key"]
    processor = model_bundle["processor"]

    if image.mode != "RGB":
        image = image.convert("RGB")

    img_array = np.array(image)
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    device = next(model.parameters()).device

    def predict_fn(images):
        pil_images = [Image.fromarray(img.astype(np.uint8)) for img in images]
        inputs = processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.sigmoid(logits).cpu().numpy()
        return pred

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_array.astype("double"),
        predict_fn,
        top_labels=5,
        hide_color=0,
        num_samples=200,
    )

    return explanation


def shap_predict_xray(image, model_bundle, results):
    """SHAP explainability for the two HF snapshot X-Ray models using Image masker."""
    model = model_bundle["model"]
    model_key = model_bundle["model_key"]
    processor = model_bundle["processor"]

    if image.mode != "RGB":
        image = image.convert("RGB")

    device = next(model.parameters()).device
    labels = _get_xray_labels(model_bundle)

    top_label_name = results[0][0] if results else labels[0]
    try:
        top_idx = labels.index(top_label_name)
    except ValueError:
        top_idx = 0

    if model_key in ("vit_chest_xray", "chexpert_convnextv2_tiny_384"):
        img_resized_rgb = np.array(image.resize((224, 224))).astype(np.float32)
        img_input = (img_resized_rgb / 255.0).astype(np.float32)

        def predict_fn(x):
            x_uint8 = np.clip(x * 255.0, 0, 255).astype(np.uint8)
            pil_images = [Image.fromarray(arr) for arr in x_uint8]
            inputs = processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
                pred = torch.sigmoid(logits).cpu().numpy()
            return pred

        masker = shap.maskers.Image("blur(16,16)", img_input.shape)
        explainer = shap.Explainer(predict_fn, masker, output_names=labels)
        shap_exp = explainer(img_input[np.newaxis, :], max_evals=300, batch_size=10)
        return shap_exp, img_resized_rgb[:, :, 0], top_idx, labels

    raise ValueError(f"Unsupported model_key for SHAP: {model_key}")


def display_xray_results(results, image, model_bundle, model_inputs, img_processed):
    """Display X-Ray analysis results"""
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Top 10 Detected Pathologies")

        top_10 = results[:10]
        df = pd.DataFrame(top_10, columns=['Pathology', 'Score'])
        df['Score (%)'] = (df['Score'] * 100).round(2)

        for idx, row in df.iterrows():
            pathology = row['Pathology']
            score = row['Score']
            score_pct = row['Score (%)']

            if score > 0.5:
                indicator = "[HIGH]"
            elif score > 0.3:
                indicator = "[MODERATE]"
            elif score > 0.1:
                indicator = "[LOW]"
            else:
                indicator = "[MINIMAL]"

            st.write(f"**{pathology}**: {score_pct}% {indicator}")
            st.progress(max(0.0, min(float(score), 1.0)))

    with col2:
        st.write("### Clinical Interpretation")

        high_risk = [r for r in results if r[1] > 0.4]

        if high_risk:
            st.warning("**Pathologies requiring attention:**")
            for path, score in high_risk[:5]:
                st.write(f"- **{path}**: {score*100:.1f}%")
            st.info("These results are provided for informational purposes only. Please consult a healthcare professional.")
        else:
            st.success("No major pathologies detected with high confidence scores.")

        st.write("### Preprocessed Image")
        if img_processed is not None:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(img_processed[0], cmap='gray')
            ax.axis('off')
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Preprocessed view is not available for this model. The model uses its own internal image processor.")

# ============== APPLICATION PAGES ==============

def homepage_audio():
    """Audio deepfake detection homepage"""
    st.write('___')
    st.subheader("Choose a wav file")
    uploaded_file = st.file_uploader(' ', type='wav', key='audio_uploader')

    if uploaded_file is not None:  
        st.write('### Play audio')
        audio_bytes = uploaded_file.getvalue()
        st.audio(audio_bytes, format='audio/wav')

        st.write('### Spectrogram Image:')
        audio_path = save_audio_bytes(uploaded_file.name, audio_bytes)

        with st.spinner('Fetching Results...'):
            spec = create_spectrogram(audio_path)
            model = tf.keras.models.load_model('saved_model/model')

        st.write('### Classification results:')
        class_label, prediction = predictions_audio(spec, model)
        st.write("#### The uploaded audio file is " + audio_class_names[class_label])

        if st.button('Show XAI Metrics', key='xai_audio'):
            st.write('### XAI Metrics using LIME')
            show_xai_note("LIME", "audio")
            with st.spinner('Fetching Results...'):
                fig2 = lime_predict_audio(spec, model)
                st.pyplot(fig2)
                plt.close(fig2)

            st.write('### XAI Metrics using Grad-CAM')
            show_xai_note("Grad-CAM", "audio")
            with st.spinner('Fetching Results...'):
                grad_fig = grad_predict_audio(spec, model, class_label)
                if grad_fig is not None:
                    st.pyplot(grad_fig)
                    plt.close(grad_fig)

            st.write('### XAI Metrics using SHAP')
            show_xai_note("SHAP", "audio")
            with st.spinner('Computing SHAP values (this may take a moment)...'):
                try:
                    shap_fig = shap_predict_audio(spec, model)
                    st.pyplot(shap_fig)
                    plt.close(shap_fig)
                    st.info("SHAP visualization shows which regions of the spectrogram contribute most to the prediction. Brighter areas indicate higher importance.")
                except Exception as e:
                    st.error(f"SHAP Error: {str(e)}")
    else:
        st.info("Please upload a .wav file")


def homepage_xray():
    """X-Ray lung pathology detection homepage"""
    st.write('___')
    st.subheader("Choose an X-Ray image")

    xray_model_choice = st.selectbox(
        "Model",
        [
            "CheXpert-5 ConvNeXtV2 Tiny (384)",
            "ViT Chest X-Ray",
        ],
        key="xray_model_choice",
    )
    if xray_model_choice.startswith("CheXpert-5 ConvNeXtV2"):
        model_key = "chexpert_convnextv2_tiny_384"
    else:
        model_key = "vit_chest_xray"

    uploaded_file = st.file_uploader(
        ' ',
        type=['png', 'jpg', 'jpeg'],
        key='xray_uploader'
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)
        with col1:
            st.write("### Original X-Ray Image")
            st.image(image, use_column_width=True)

        with st.spinner('Loading X-Ray model...'):
            model_bundle = load_xray_model(model_key)

        if model_bundle is None or model_bundle.get("model") is None:
            st.error("Failed to load X-Ray model")
            return

        with st.spinner('Analyzing image...'):
            results, model_inputs, img_processed = predictions_xray(image, model_bundle)

        with col2:
            st.write("### Preprocessed Image")
            if img_processed is not None:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(img_processed[0], cmap='gray')
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Preprocessed view is not available for this model.")

        st.write('---')
        display_xray_results(results, image, model_bundle, model_inputs, img_processed)

        if st.button('Show XAI Metrics', key='xai_xray'):
            st.write('---')

            if model_bundle["model_key"] == "chexpert_convnextv2_tiny_384":
                st.write('### XAI Metrics using Grad-CAM')
                show_xai_note("Grad-CAM", "xray", model_bundle["model_key"])
                with st.spinner('Computing Grad-CAM...'):
                    try:
                        pixel_values = model_inputs.get("pixel_values")
                        if pixel_values is None:
                            raise ValueError("Missing pixel_values in model inputs.")

                        cam = grad_cam_hf_image_model(model_bundle["model"], pixel_values)
                        if cam is None:
                            st.warning("Unable to generate Grad-CAM for this model.")
                        else:
                            cam_224 = cv2.resize(cam.astype(np.float32), (224, 224))
                            img_gray = np.array(image.convert('L'))
                            img_gray = cv2.resize(img_gray, (224, 224))

                            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                            axes[0].imshow(img_gray, cmap='gray')
                            axes[0].set_title("Original Image")
                            axes[0].axis('off')

                            axes[1].imshow(cam_224, cmap='jet')
                            axes[1].set_title("Grad-CAM Activation Map")
                            axes[1].axis('off')

                            axes[2].imshow(img_gray, cmap='gray')
                            axes[2].imshow(cam_224, cmap='jet', alpha=0.5)
                            axes[2].set_title("Superimposed (Regions of Interest)")
                            axes[2].axis('off')

                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()

                            st.info("Red/yellow regions indicate areas the model considers important for its prediction.")
                    except Exception as e:
                        st.error(f"Grad-CAM Error: {str(e)}")
            else:
                st.write('### XAI Metrics using Attention Rollout (ViT)')
                show_xai_note("Attention Rollout", "xray", model_bundle["model_key"])
                with st.spinner('Computing Attention Rollout...'):
                    try:
                        heatmap = vit_attention_rollout_xray(model_bundle, image)

                        img_gray = np.array(image.convert('L'))
                        img_gray = cv2.resize(img_gray, (224, 224))

                        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                        axes[0].imshow(img_gray, cmap='gray')
                        axes[0].set_title("Original Image")
                        axes[0].axis('off')

                        axes[1].imshow(heatmap, cmap='jet')
                        axes[1].set_title("Attention Rollout Map")
                        axes[1].axis('off')

                        axes[2].imshow(img_gray, cmap='gray')
                        axes[2].imshow(heatmap, cmap='jet', alpha=0.5)
                        axes[2].set_title("Superimposed (Regions of Interest)")
                        axes[2].axis('off')

                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

                        st.info(
                            "For ViT-based models, an attention-based heatmap is provided instead of Grad-CAM."
                        )
                    except Exception as e:
                        st.error(f"Attention Rollout Error: {str(e)}")

            st.write('### XAI Metrics using LIME')
            show_xai_note("LIME", "xray", model_bundle["model_key"])
            with st.spinner('Computing LIME (this may take a few minutes)...'):
                try:
                    explanation = lime_predict_xray(image, model_bundle)

                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                    axes[0].imshow(image, cmap='gray')
                    axes[0].set_title("Original Image")
                    axes[0].axis('off')

                    top_label = explanation.top_labels[0]
                    labels = _get_xray_labels(model_bundle)

                    temp, mask = explanation.get_image_and_mask(
                        top_label,
                        positive_only=True,
                        num_features=10,
                        hide_rest=True
                    )
                    temp_normalized = np.clip(temp / 255.0, 0, 1) if temp.max() > 1 else np.clip(temp, 0, 1)
                    axes[1].imshow(mark_boundaries(temp_normalized, mask))
                    axes[1].set_title(f"Positive Regions - {labels[top_label]}")
                    axes[1].axis('off')

                    temp2, mask2 = explanation.get_image_and_mask(
                        top_label,
                        positive_only=False,
                        num_features=10,
                        hide_rest=False
                    )
                    temp2_normalized = np.clip(temp2 / 255.0, 0, 1) if temp2.max() > 1 else np.clip(temp2, 0, 1)
                    axes[2].imshow(mark_boundaries(temp2_normalized, mask2))
                    axes[2].set_title("Positive (green) and Negative (red) Regions")
                    axes[2].axis('off')

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    st.info("Green regions contribute positively to the prediction. Red regions contribute negatively.")
                except Exception as e:
                    st.error(f"LIME Error: {str(e)}")

            st.write('### XAI Metrics using SHAP')
            show_xai_note("SHAP", "xray", model_bundle["model_key"])
            with st.spinner('Computing SHAP values (this may take several minutes)...'):
                try:
                    shap_values, img_resized, top_idx, labels = shap_predict_xray(image, model_bundle, results)

                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                    axes[0].imshow(img_resized, cmap='gray')
                    axes[0].set_title("Original Image")
                    axes[0].axis('off')

                    shap_val = shap_values.values[0, :, :, 0, top_idx]

                    axes[1].imshow(img_resized, cmap='gray')
                    pos_shap = np.maximum(shap_val, 0)
                    axes[1].imshow(pos_shap, cmap='Reds', alpha=0.7)
                    axes[1].set_title(f"SHAP Positive - {labels[top_idx]}")
                    axes[1].axis('off')

                    axes[2].imshow(img_resized, cmap='gray')
                    axes[2].imshow(shap_val, cmap='seismic', alpha=0.7,
                                   vmin=-np.abs(shap_val).max(), vmax=np.abs(shap_val).max())
                    axes[2].set_title("SHAP Values (Red=Positive, Blue=Negative)")
                    axes[2].axis('off')

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    st.info("SHAP values show feature importance: red regions contribute positively to the prediction, blue regions contribute negatively.")
                except Exception as e:
                    st.error(f"SHAP Error: {str(e)}")
    else:
        st.info("Please upload an X-Ray image (PNG, JPG, JPEG)")


def unified_page():
    """Single unified workflow: one upload -> auto modality -> filtered models + XAI + comparison tabs."""
    st.write('___')
    st.subheader("Upload file")
    uploaded = st.file_uploader(" ", type=["wav", "png", "jpg", "jpeg"], key="unified_uploader")

    if uploaded is None:
        st.info("Upload a .wav file for audio analysis or an image (png/jpg/jpeg) for X-Ray analysis.")
        return

    name = uploaded.name.lower()
    is_audio = name.endswith(".wav")

    if is_audio:
        st.write("### Modality: Audio (.wav)")
        audio_bytes = uploaded.getvalue()
        st.audio(audio_bytes, format='audio/wav')

        audio_path = save_audio_bytes(uploaded.name, audio_bytes)
        file_id = _sha1_bytes(audio_bytes)

        with st.spinner("Creating spectrogram and loading model..."):
            spec = create_spectrogram(audio_path)
            model = tf.keras.models.load_model('saved_model/model')

        class_label, prediction = predictions_audio(spec, model)

        if hasattr(st, "tabs"):
            tab_pred, tab_xai, tab_cmp = st.tabs(["Prediction", "Explain", "Compare"])
        else:
            tab_choice = st.radio("View", ["Prediction", "Explain", "Compare"], horizontal=True, key="unified_audio_view")
            tab_pred = tab_xai = tab_cmp = None

        def _show_audio_pred():
            st.write("### Classification results")
            st.write("#### The uploaded audio file is " + audio_class_names[class_label])

        # Compatibility filtering
        available_xai = ["LIME", "SHAP", "Grad-CAM"]
        chosen = st.multiselect("XAI methods", available_xai, default=["LIME", "Grad-CAM"], key="unified_audio_xai")

        def _show_audio_explain():
            if "LIME" in chosen:
                st.write("### LIME")
                show_xai_note("LIME", "audio")
                fig = _xai_cached(f"audio:{file_id}:lime", lambda: lime_predict_audio(spec, model))
                if fig is not None:
                    st.pyplot(fig)
            if "Grad-CAM" in chosen:
                st.write("### Grad-CAM")
                show_xai_note("Grad-CAM", "audio")
                fig = _xai_cached(f"audio:{file_id}:gradcam", lambda: grad_predict_audio(spec, model, class_label))
                if fig is not None:
                    st.pyplot(fig)
            if "SHAP" in chosen:
                st.write("### SHAP")
                show_xai_note("SHAP", "audio")
                fig = _xai_cached(f"audio:{file_id}:shap", lambda: shap_predict_audio(spec, model))
                if fig is not None:
                    st.pyplot(fig)

        def _show_audio_compare():
            st.write("### Comparison")
            st.caption(f"Prediction: {audio_class_names[class_label]}")
            cols = st.columns(max(1, len(chosen)))
            for i, method in enumerate(chosen):
                with cols[i]:
                    st.write(f"**{method}**")
                    if method == "LIME":
                        fig = _xai_cached(f"audio:{file_id}:lime", lambda: lime_predict_audio(spec, model))
                        if fig is not None:
                            st.pyplot(fig)
                    elif method == "Grad-CAM":
                        fig = _xai_cached(f"audio:{file_id}:gradcam", lambda: grad_predict_audio(spec, model, class_label))
                        if fig is not None:
                            st.pyplot(fig)
                    elif method == "SHAP":
                        fig = _xai_cached(f"audio:{file_id}:shap", lambda: shap_predict_audio(spec, model))
                        if fig is not None:
                            st.pyplot(fig)

        if hasattr(st, "tabs"):
            with tab_pred:
                _show_audio_pred()
            with tab_xai:
                _show_audio_explain()
            with tab_cmp:
                _show_audio_compare()
        else:
            if tab_choice == "Prediction":
                _show_audio_pred()
            elif tab_choice == "Explain":
                _show_audio_explain()
            else:
                _show_audio_compare()
        return

    # Image / X-Ray
    st.write("### Modality: X-Ray Image")
    image_bytes = uploaded.getvalue()
    file_id = _sha1_bytes(image_bytes)
    image = Image.open(uploaded)
    st.image(image, use_column_width=True)

    # Model selector filtered for image modality
    xray_choice = st.selectbox(
        "Model",
        ["CheXpert-5 ConvNeXtV2 Tiny (384)", "ViT Chest X-Ray"],
        key="unified_xray_model",
    )
    model_key = "chexpert_convnextv2_tiny_384" if xray_choice.startswith("CheXpert-5 ConvNeXtV2") else "vit_chest_xray"

    with st.spinner("Loading model and running inference..."):
        model_bundle = load_xray_model(model_key)
        results, model_inputs, img_processed = predictions_xray(image, model_bundle)

    if hasattr(st, "tabs"):
        tab_pred, tab_xai, tab_cmp = st.tabs(["Prediction", "Explain", "Compare"])
    else:
        tab_choice = st.radio("View", ["Prediction", "Explain", "Compare"], horizontal=True, key="unified_xray_view")
        tab_pred = tab_xai = tab_cmp = None

    def _show_xray_pred():
        display_xray_results(results, image, model_bundle, model_inputs, img_processed)

    # Compatibility filtering based on model architecture
    if model_key == "chexpert_convnextv2_tiny_384":
        available_xai = ["LIME", "SHAP", "Grad-CAM"]
        default_xai = ["Grad-CAM", "LIME"]
    else:
        available_xai = ["LIME", "SHAP", "Attention Rollout"]
        default_xai = ["Attention Rollout", "LIME"]

    chosen = st.multiselect("XAI methods", available_xai, default=default_xai, key="unified_xray_xai")

    def _show_xray_explain():
        if "Grad-CAM" in chosen:
            st.write("### Grad-CAM")
            show_xai_note("Grad-CAM", "xray", model_key)
            pv = model_inputs.get("pixel_values")
            cam = _xai_cached(
                f"xray:{file_id}:{model_key}:gradcam",
                lambda: (grad_cam_hf_image_model(model_bundle["model"], pv) if pv is not None else None),
            )
            if cam is None:
                st.warning("Unable to generate Grad-CAM for this model.")
            else:
                cam_224 = cv2.resize(cam.astype(np.float32), (224, 224))
                img_gray = cv2.resize(np.array(image.convert("L")), (224, 224))
                fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                ax[0].imshow(img_gray, cmap="gray"); ax[0].axis("off"); ax[0].set_title("Input")
                ax[1].imshow(img_gray, cmap="gray"); ax[1].imshow(cam_224, cmap="jet", alpha=0.5); ax[1].axis("off"); ax[1].set_title("Overlay")
                st.pyplot(fig); plt.close()
        if "Attention Rollout" in chosen:
            st.write("### Attention Rollout (ViT)")
            show_xai_note("Attention Rollout", "xray", model_key)
            heatmap = _xai_cached(
                f"xray:{file_id}:{model_key}:attention",
                lambda: vit_attention_rollout_xray(model_bundle, image),
            )
            img_gray = cv2.resize(np.array(image.convert("L")), (224, 224))
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].imshow(img_gray, cmap="gray"); ax[0].axis("off"); ax[0].set_title("Input")
            ax[1].imshow(img_gray, cmap="gray"); ax[1].imshow(heatmap, cmap="jet", alpha=0.5); ax[1].axis("off"); ax[1].set_title("Overlay")
            st.pyplot(fig); plt.close()
        if "LIME" in chosen:
            st.write("### LIME")
            show_xai_note("LIME", "xray", model_key)
            explanation = _xai_cached(
                f"xray:{file_id}:{model_key}:lime",
                lambda: lime_predict_xray(image, model_bundle),
            )
            top_label = explanation.top_labels[0]
            labels = _get_xray_labels(model_bundle)
            temp, mask = explanation.get_image_and_mask(top_label, positive_only=True, num_features=10, hide_rest=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(mark_boundaries(np.clip(temp / 255.0, 0, 1), mask))
            ax.set_title(labels[top_label] if top_label < len(labels) else f"Class {top_label}")
            ax.axis("off")
            st.pyplot(fig); plt.close()
        if "SHAP" in chosen:
            st.write("### SHAP")
            show_xai_note("SHAP", "xray", model_key)
            shap_values, img_resized, top_idx, labels = _xai_cached(
                f"xray:{file_id}:{model_key}:shap",
                lambda: shap_predict_xray(image, model_bundle, results),
            )
            shap_val = shap_values.values[0, :, :, 0, top_idx]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.imshow(img_resized, cmap="gray")
            ax.imshow(shap_val, cmap="seismic", alpha=0.6, vmin=-np.abs(shap_val).max(), vmax=np.abs(shap_val).max())
            ax.axis("off")
            st.pyplot(fig); plt.close()

    def _show_xray_compare():
        st.write("### Comparison")
        if results:
            st.caption(f"Top prediction: {results[0][0]} ({results[0][1]*100:.1f}%)")
        cols = st.columns(max(1, len(chosen)))
        for i, method in enumerate(chosen):
            with cols[i]:
                st.write(f"**{method}**")
                if method == "Grad-CAM":
                    pv = model_inputs.get("pixel_values")
                    cam = _xai_cached(
                        f"xray:{file_id}:{model_key}:gradcam",
                        lambda: (grad_cam_hf_image_model(model_bundle["model"], pv) if pv is not None else None),
                    )
                    if cam is not None:
                        st.image(cv2.resize(cam.astype(np.float32), (224, 224)))
                elif method == "Attention Rollout":
                    st.image(_xai_cached(
                        f"xray:{file_id}:{model_key}:attention",
                        lambda: vit_attention_rollout_xray(model_bundle, image),
                    ))
                elif method == "LIME":
                    explanation = _xai_cached(
                        f"xray:{file_id}:{model_key}:lime",
                        lambda: lime_predict_xray(image, model_bundle),
                    )
                    top_label = explanation.top_labels[0]
                    temp, mask = explanation.get_image_and_mask(top_label, positive_only=True, num_features=10, hide_rest=True)
                    st.image(mark_boundaries(np.clip(temp / 255.0, 0, 1), mask))
                elif method == "SHAP":
                    shap_values, img_resized, top_idx, labels = _xai_cached(
                        f"xray:{file_id}:{model_key}:shap",
                        lambda: shap_predict_xray(image, model_bundle, results),
                    )
                    shap_val = shap_values.values[0, :, :, 0, top_idx]
                    vmax = float(np.abs(shap_val).max()) if np.abs(shap_val).max() > 0 else 1.0
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.imshow(img_resized, cmap="gray")
                    ax.imshow(shap_val, cmap="seismic", alpha=0.6, vmin=-vmax, vmax=vmax)
                    ax.axis("off")
                    st.pyplot(fig)
                    plt.close(fig)

    if hasattr(st, "tabs"):
        with tab_pred:
            _show_xray_pred()
        with tab_xai:
            _show_xray_explain()
        with tab_cmp:
            _show_xray_compare()
    else:
        if tab_choice == "Prediction":
            _show_xray_pred()
        elif tab_choice == "Explain":
            _show_xray_explain()
        else:
            _show_xray_compare()


def about():
    """About page"""
    st.title("About present work")

    st.markdown("""
    ### Audio Deepfake Detection
    
    This application detects deepfake audio using a TensorFlow/Keras CNN classifier trained on MEL spectrograms. 
    Audio files (.wav) are converted to spectrograms (224x224 images) which are then classified as **real** or **fake**.
    
    The audio detection pipeline is based on the work from [Deepfake-Audio-Detection-with-XAI](https://github.com/Guri10/Deepfake-Audio-Detection-with-XAI), 
    which explores multiple architectures (VGG16, MobileNet, ResNet, Custom CNN) for deepfake audio classification.
    
    XAI methods (LIME, Grad-CAM, SHAP) are applied to the spectrogram to explain which frequency/time regions 
    influence the model's decision.
    """)

    st.write("---")

    st.markdown("""
    ### Lung Pathology Detection
    
    This project includes chest X-Ray analysis for lung pathology detection using two Hugging Face models:
    
    - **CheXpert-5 ConvNeXtV2 Tiny (384)**: ConvNeXtV2 architecture fine-tuned on CheXpert dataset. 
      Detects 5 pathologies: Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion.
      XAI methods: LIME, SHAP, Grad-CAM.
    
    - **ViT Chest X-Ray**: Vision Transformer fine-tuned on CheXpert dataset.
      Detects 5 pathologies: Cardiomegaly, Edema, Consolidation, Pneumonia, No Finding.
      XAI methods: LIME, SHAP, Attention Rollout (Grad-CAM not applicable to transformers).
    
    Models are loaded from local offline snapshots for reliability during demonstrations.
    """)

    st.write("---")

    st.markdown("""
    ### XAI Methods Used

    - **LIME** (Local Interpretable Model-agnostic Explanations): Provides local explanations by perturbing the input and observing the impact on predictions.
    - **Grad-CAM** (Gradient-weighted Class Activation Mapping): Visualizes the regions of an image that are most important for a specific prediction.
    - **SHAP** (SHapley Additive exPlanations): Uses game-theoretic approach to explain predictions by computing the contribution of each feature.
    """)

    st.write("---")

    st.markdown("""
    ### AI Usage in Project Realization

    We used cursor ai with Opus 4.5 to help us to correct our code after a first implementation of our solution to improve our project.
    We also used ChatGPT 5.2 to help us to understand the project and to give feedback on our code and our report for mistakes we made and could miss after reviewing our project.
    

    """)

    st.write("---")

    st.markdown("""
    ### Disclaimer

    These tools are provided for educational and research purposes only. The lung pathology detection results **do not replace professional medical diagnosis** and should be interpreted by qualified healthcare professionals.
    """)


def main():
    """Main application function"""
    page = st.sidebar.selectbox(
        "App Selections",
        ["Unified", "Deepfake Audio", "X-Ray Lung Pathology", "About"]
    )

    if page == "Unified":
        st.title("Unified XAI Interface")
        unified_page()
    elif page == "Deepfake Audio":
        st.title("Deepfake Audio Detection using XAI")
        homepage_audio()
    elif page == "X-Ray Lung Pathology":
        st.title("Lung Pathology Detection using XAI")
        st.markdown("*Offline Hugging Face models: ConvNeXtV2 CheXpert-5 and ViT Chest X-Ray*")
        homepage_xray()
    elif page == "About":
        about()


if __name__ == "__main__":
    main()
