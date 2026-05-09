import base64
import io
import sys
import os

# Ensure src modules can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torchvision.transforms as T
from PIL import Image

from add_adversarial_noise import AdversarialImageGenerator, preprocess_image
from remove_adversarial_noise import AdversarialDenoiser, compute_psnr, compute_ssim
from detect_adversarial import AdversarialDetector
from utils import denormalize, MEAN, STANDARD_DEVIATION, get_class_values_from_idx, load_model

# Shared detector (heuristic is stateless; ANN training happens per-session)
_detector = AdversarialDetector()

app = FastAPI(title="Adversarial Attack Tool")

# Allow CORS for React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CustomAdversarialImageGenerator(AdversarialImageGenerator):
    """Subclassing to inject the configurable epsilon."""
    def __init__(self, target_class: str, epsilon: float):
        super().__init__(target_class)
        self.epsilon = epsilon

    def iterative_target_class_method(self, input_img, data_grad, epsilon=0.25, alpha=0.025):
        # Override epsilon with the user-provided one
        return super().iterative_target_class_method(input_img, data_grad, epsilon=self.epsilon, alpha=alpha)


def get_top5_predictions(model, img: torch.Tensor):
    output = model(img)
    probabilities = torch.softmax(output, dim=1)[0]
    top5_probs, top5_indices = torch.topk(probabilities, 5)

    predictions = []
    for prob, idx in zip(top5_probs, top5_indices):
        label = get_class_values_from_idx(idx.item())
        # The labels might be comma separated "gibbon, Hylobates lar", just taking the first part for cleaner UI
        label_short = label.split(",")[0]
        predictions.append({"label": label_short, "confidence": float(prob.item() * 100)})
    return predictions


def tensor_to_base64(tensor: torch.Tensor, is_noise=False) -> str:
    tensor = tensor.detach().cpu()
    if is_noise:
        # Amplify noise to make it visible: min-max normalization
        min_v = tensor.min()
        max_v = tensor.max()
        if max_v - min_v > 1e-8:
            tensor = (tensor - min_v) / (max_v - min_v)
        else:
            tensor = torch.zeros_like(tensor)
    else:
        # Denormalize the image back to [0, 1] range
        tensor = denormalize(tensor, mean=MEAN, std=STANDARD_DEVIATION)
        tensor = torch.clamp(tensor, 0, 1)

    img_pil = T.ToPILImage()(tensor.squeeze(0))
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")


@app.post("/attack")
async def perform_attack(
    image: UploadFile = File(...),
    target_class: str = Form(...),
    epsilon: float = Form(0.05),
    iterations: int = Form(5),
):
    try:
        # Load and preprocess the image
        img_bytes = await image.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        input_tensor = preprocess_image(pil_img)

        # Initialize the generator
        generator = CustomAdversarialImageGenerator(target_class=target_class, epsilon=epsilon)

        # Generate adversarial image (this takes care of clipping and generating noise)
        result = generator.generate_adversarial_image(input_tensor, max_iterations=iterations)
        
        # Get baseline stats
        orig_top5 = get_top5_predictions(generator.model, input_tensor)
        orig_img_b64 = tensor_to_base64(input_tensor)

        if result.adv_img is None:
            # The attack failed to reach the target class within iterations
            # We'll just return the current progress or fail gracefully
            # Wait, `generate_adversarial_image` returns an empty Result if it fails.
            # But the question asks to show a clear warning message if it fails.
            # To show a warning, we still need the latest adversarial image if possible, 
            # but generate_adversarial_image doesn't save it. We'll have to just return the error.
            return {"success": False, "message": "Failed to fool the model. Try increasing epsilon or iterations."}

        adv_tensor = result.adv_img
        noise_tensor = adv_tensor - input_tensor

        # Get stats for adversarial image
        adv_top5 = get_top5_predictions(generator.model, adv_tensor)
        
        # Build base64 outputs
        adv_img_b64 = tensor_to_base64(adv_tensor)
        noise_img_b64 = tensor_to_base64(noise_tensor, is_noise=True)

        return {
            "success": True,
            "original_prediction": {"label": result.orig_label.split(",")[0], "confidence": result.orig_prob},
            "adversarial_prediction": {"label": result.adv_label.split(",")[0], "confidence": result.adv_prob},
            "original_top5": orig_top5,
            "adversarial_top5": adv_top5,
            "images": {
                "original": orig_img_b64,
                "adversarial": adv_img_b64,
                "noise": noise_img_b64
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------------
# /denoise endpoint — adversarial noise removal
# ---------------------------------------------------------------------------

# Shared model instance for the denoise endpoint (avoids reloading weights)
_shared_model = None

def _get_model():
    global _shared_model
    if _shared_model is None:
        _shared_model = load_model()
    return _shared_model


@app.post("/denoise")
async def perform_denoise(
    image: UploadFile = File(...),
    method: str = Form("tv"),
    epsilon: float = Form(0.05),
):
    """
    Accepts an (adversarial) image and applies the selected denoising defense.

    Returns:
        denoised_image       : base64 PNG of the denoised result
        original_prediction  : top-1 label + confidence BEFORE denoising
        restored_prediction  : top-1 label + confidence AFTER denoising
        psnr_score           : Peak Signal-to-Noise Ratio (higher = better)
        ssim_score           : Structural Similarity Index (1.0 = identical)
        success              : True when the top-1 prediction changed
    """
    try:
        # ── 1. Load and preprocess the uploaded image ──────────────────────
        img_bytes  = await image.read()
        pil_img    = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        input_tensor = preprocess_image(pil_img)   # [1, 3, 224, 224] normalized

        model = _get_model()

        # ── 2. Get the model's prediction BEFORE denoising ─────────────────
        with torch.no_grad():
            orig_output  = model(input_tensor)
        orig_probs       = torch.softmax(orig_output, dim=1)[0]
        orig_conf, orig_idx = orig_probs.max(0)
        orig_label       = get_class_values_from_idx(orig_idx.item()).split(",")[0]
        orig_confidence  = float(orig_conf.item() * 100)

        # ── 3. Apply the chosen denoising defense ──────────────────────────
        denoiser      = AdversarialDenoiser(epsilon=epsilon)
        denoised_tensor = denoiser.denoise(input_tensor, method=method)

        # ── 4. Get the model's prediction AFTER denoising ──────────────────
        with torch.no_grad():
            rest_output  = model(denoised_tensor)
        rest_probs       = torch.softmax(rest_output, dim=1)[0]
        rest_conf, rest_idx = rest_probs.max(0)
        rest_label       = get_class_values_from_idx(rest_idx.item()).split(",")[0]
        rest_confidence  = float(rest_conf.item() * 100)

        # ── 5. Compute image quality metrics ───────────────────────────────
        mean_t = denoiser.mean
        std_t  = denoiser.std
        psnr   = compute_psnr(input_tensor, denoised_tensor, mean_t, std_t)
        ssim   = compute_ssim(input_tensor, denoised_tensor, mean_t, std_t)

        # ── 6. Encode the denoised image to base64 ─────────────────────────
        denoised_b64 = tensor_to_base64(denoised_tensor)

        return {
            "denoised_image"      : denoised_b64,
            "original_prediction" : {"label": orig_label, "confidence": orig_confidence},
            "restored_prediction" : {"label": rest_label, "confidence": rest_confidence},
            "psnr_score"          : round(psnr, 3),
            "ssim_score"          : round(ssim, 4),
            # success = True when top-1 class changed after denoising
            "success"             : orig_idx.item() != rest_idx.item(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# /download endpoint — reliable file download with proper filename
# ---------------------------------------------------------------------------

from fastapi.responses import StreamingResponse
from pydantic import BaseModel as PydanticBase

class DownloadRequest(PydanticBase):
    data_url: str   # "data:image/png;base64,..."
    filename: str   # e.g. "adversarial_gibbon.png"

@app.post("/download")
async def download_image(req: DownloadRequest):
    """
    Accepts a base64 data URL and returns it as a file download.

    This is the most reliable cross-browser way to force a proper filename:
    the browser ALWAYS respects Content-Disposition: attachment; filename=...
    regardless of Chrome's restrictions on client-side data:/blob: URLs.
    """
    try:
        # Strip the data URL prefix (data:image/png;base64,<data>)
        if "," not in req.data_url:
            raise HTTPException(status_code=400, detail="Invalid data URL")

        header, b64_data = req.data_url.split(",", 1)

        # Determine MIME type from header (e.g. "data:image/png;base64")
        mime = "image/png"
        if ":" in header and ";" in header:
            mime = header.split(":")[1].split(";")[0]

        # Sanitise filename — strip path traversal, ensure .png extension
        safe_filename = os.path.basename(req.filename) or "image.png"
        if not safe_filename.lower().endswith((".png", ".jpg", ".jpeg")):
            safe_filename += ".png"

        img_bytes = base64.b64decode(b64_data)

        return StreamingResponse(
            io.BytesIO(img_bytes),
            media_type=mime,
            headers={
                "Content-Disposition": f'attachment; filename="{safe_filename}"',
                "Content-Length": str(len(img_bytes)),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# /detect endpoint — adversarial image detection (Feature Squeezing method)
# ---------------------------------------------------------------------------


@app.post("/detect")
async def detect_adversarial(
    image: UploadFile = File(...),
    epsilon: float = Form(0.05),
):
    """
    Detect adversarial perturbations using the Feature Squeezing method.

    Applies 3 squeezers (Gaussian blur, bit-depth reduction, JPEG) and
    measures how much the model's softmax output shifts.  A large shift
    indicates the image is adversarial (perturbations are brittle).

    Returns:
        is_adversarial  bool
        confidence      float 0-1
        verdict         str
        votes           int  (0-3: how many squeezers triggered)
        prediction      dict (label + confidence)
        scores          dict (per-squeezer L1 delta + max)
        features        list [d_blur, d_bits, d_jpeg]
        threshold       float (threshold used for this epsilon)
    """
    try:
        img_bytes    = await image.read()
        pil_img      = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        input_tensor = preprocess_image(pil_img)

        # Run Feature Squeezing detector
        result = _detector.detect(input_tensor, epsilon=epsilon, use_ann=False)

        # Get ResNet's current top-1 prediction for context
        model = _get_model()
        with torch.no_grad():
            output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        conf, idx = probs.max(0)
        label = get_class_values_from_idx(idx.item()).split(",")[0]

        return {
            "is_adversarial" : result["is_adversarial"],
            "confidence"     : result["confidence"],
            "votes"          : result["votes"],
            "verdict"        : result["verdict"],
            "prediction"     : {"label": label, "confidence": round(float(conf.item() * 100), 2)},
            "scores"         : result["scores"],
            "features"       : result["features"],
            "threshold"      : result["threshold"],
            "delta_blur"     : result["delta_blur"],
            "delta_bits"     : result["delta_bits"],
            "delta_jpeg"     : result["delta_jpeg"],
            "delta_max"      : result["delta_max"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/classes")
def get_classes():
    """Endpoint to get all 1000 ImageNet classes for the frontend dropdown."""
    import json
    from utils import IMAGENET_CLASS_IDX_VALUE_MAPPING
    mapping_path = os.path.join(os.path.dirname(__file__), "src", IMAGENET_CLASS_IDX_VALUE_MAPPING)
    with open(mapping_path, "r") as f:
        data = json.load(f)
    classes = []
    for key, value in data.items():
        # Clean up commas and extra spaces
        name = value.split(",")[0].strip()
        classes.append({"id": key, "name": name, "raw": value})
    classes.sort(key=lambda x: x["name"])
    return classes

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
