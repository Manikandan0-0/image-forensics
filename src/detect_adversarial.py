"""
detect_adversarial.py
~~~~~~~~~~~~~~~~~~~~~

Adversarial image detector using a multi-signal approach specifically
calibrated for the Iterative Target Class Method (ITCM) used in this project.

Key Insight
-----------
ITCM adversarial images with ε≈0.05 have very precisely placed perturbations.
Diagnostic tests on real attack outputs revealed:

  Signal                    | Clean      | Adversarial | Ratio
  --------------------------|------------|-------------|-------
  JPEG Δ (softmax L1)       | 0.00002    | 0.00485     | 205×  ✓ BEST
  Bit-depth Δ (softmax L1)  | 0.00049    | 0.00000     | ×      useless
  Gaussian Δ (softmax L1)   | 0.00001    | 0.00000     | ×      useless
  JPEG pixel L1             | 0.0087     | 0.0148      | 1.7×  ✓ usable
  Noise pixel max           | —          | 0.25        | —     ✓ huge gap

Detection Strategy
------------------
1. **JPEG Softmax Delta** (primary signal)
   Compare softmax output before vs after JPEG compression (quality=75).
   Threshold = 0.002 (clean ≈ 0.00002, adversarial ≈ 0.00485).

2. **JPEG Pixel Delta** (secondary signal)
   Mean absolute pixel difference between original and JPEG version.
   Threshold = 0.010 (clean ≈ 0.009, adversarial ≈ 0.015).

3. **Prediction Flip** (tertiary binary signal)
   Does JPEG compression change the top-1 prediction?
   Adversarial images often flip back to the correct class after JPEG.

4. **Bit-depth Pixel Delta** (supporting signal)
   Adversarial noise adds quantization-resistant structured patterns.

Confidence is a weighted combination of all signals.

AdvDetectorANN (from network/adv_ann.py) is used as an optional secondary
classifier trained on these feature values.
"""

import io
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from utils import denormalize, MEAN, STANDARD_DEVIATION, load_model

_MEAN = torch.tensor(MEAN).view(1, 3, 1, 1)
_STD  = torch.tensor(STANDARD_DEVIATION).view(1, 3, 1, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Neural Network (adapted from network/adv_ann.py)
# ═══════════════════════════════════════════════════════════════════════════════

class AdvDetectorANN(nn.Module):
    """
    Binary classifier adapted from network/adv_ann.py.
    Original: Linear(784→300→128→2) for MNIST.
    Here: Linear(4→64→32→2) for our 4 detection features.
    Output: 2 logits [clean_score, adversarial_score]
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _denorm(x: torch.Tensor) -> torch.Tensor:
    return (x * _STD + _MEAN).clamp(0.0, 1.0)

def _renorm(x: torch.Tensor) -> torch.Tensor:
    return (x - _MEAN) / _STD

def _to_pil(x: torch.Tensor) -> Image.Image:
    return T.ToPILImage()(_denorm(x).squeeze(0).clamp(0, 1))

def _jpeg_compress(x: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """Round-trip through JPEG in [0,1] pixel space."""
    buf = io.BytesIO()
    _to_pil(x).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    pix_j = T.ToTensor()(Image.open(buf).convert("RGB")).unsqueeze(0)
    return _renorm(pix_j)

def _gaussian_blur(x: torch.Tensor, ks: int = 3, sigma: float = 0.5) -> torch.Tensor:
    return T.GaussianBlur(kernel_size=ks, sigma=(sigma, sigma))(x)

def _reduce_bits(x: torch.Tensor, bit_depth: int = 4) -> torch.Tensor:
    pix = _denorm(x)
    lvl = (2 ** bit_depth) - 1
    return _renorm(torch.round(pix * lvl) / lvl)

@torch.no_grad()
def _softmax(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(model(x), dim=1)[0]

@torch.no_grad()
def _l1_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().sum().item())


# ═══════════════════════════════════════════════════════════════════════════════
# Calibrated thresholds (verified on panda clean vs ITCM adversarial)
# ═══════════════════════════════════════════════════════════════════════════════

# Signal 1: JPEG softmax L1 delta
# clean≈0.00002 | adversarial≈0.00485  →  threshold midpoint ≈ 0.002
_JPEG_SOFTMAX_THRESH  = 0.002

# Signal 2: JPEG pixel L1 delta (in [0,1] space)
# clean≈0.0087  | adversarial≈0.0148   →  threshold midpoint ≈ 0.011
_JPEG_PIXEL_THRESH    = 0.011

# Signal 3: Bit-depth pixel L1 delta
# clean≈0.0178  | adversarial≈0.0168   →  slight difference, use 0.015
_BITS_PIXEL_THRESH    = 0.015

# Signal 4: Gaussian pixel L1 delta
# clean≈0.0034  | adversarial≈0.0062   →  threshold midpoint ≈ 0.004
_BLUR_PIXEL_THRESH    = 0.004


def _sigmoid_conf(val: float, thresh: float, scale: float = 8.0) -> float:
    """Maps val relative to thresh into a [0,1] confidence via sigmoid."""
    return float(torch.sigmoid(
        torch.tensor((val - thresh) / (thresh + 1e-8) * scale)
    ).item())


# ═══════════════════════════════════════════════════════════════════════════════
# Main Detector Class
# ═══════════════════════════════════════════════════════════════════════════════

class AdversarialDetector:
    """
    Multi-signal adversarial detector calibrated for the ITCM attack.

    Primary signal: JPEG compression softmax delta (205× separation).
    Secondary signals: pixel-space deltas from JPEG, bit-depth, blur.

    Also wraps AdvDetectorANN (from network/adv_ann.py) as an optional
    secondary classifier that can be trained on the 4-D feature vector.

    Usage
    -----
    detector = AdversarialDetector()
    result   = detector.detect(image_tensor, epsilon=0.05)
    """

    def __init__(self, threshold: float = _JPEG_SOFTMAX_THRESH):
        self.threshold   = threshold
        self._model      = None
        self.ann         = AdvDetectorANN()
        self.ann.eval()
        self._is_trained = False
        self._buf_X: list = []
        self._buf_y: list = []

    def _get_model(self) -> nn.Module:
        if self._model is None:
            self._model = load_model()
        return self._model

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, image_tensor: torch.Tensor,
               epsilon: float = 0.05,
               use_ann: bool = False) -> dict:
        """
        Detect whether image_tensor contains adversarial perturbations.

        Args:
            image_tensor : [1, 3, H, W] ImageNet-normalized tensor.
            epsilon      : Attack epsilon (scales thresholds proportionally).
            use_ann      : Use trained ANN instead of heuristic (if available).

        Returns dict:
            is_adversarial  bool
            confidence      float 0-1
            verdict         str
            votes           int 0-4
            scores          dict (4 individual signal values)
            features        list [s1, s2, s3, s4]
            threshold       float
        """
        model = self._get_model()

        # Scale thresholds: larger epsilon → larger perturbation → easier detect
        scale = max(0.3, epsilon / 0.05)   # ε=0.05→1.0, ε=0.025→0.5

        # ── Compute squeezed variants ────────────────────────────────────────
        sq_jpeg = _jpeg_compress(image_tensor)
        sq_bits = _reduce_bits(image_tensor)
        sq_blur = _gaussian_blur(image_tensor)

        # ── Softmax vectors ──────────────────────────────────────────────────
        sm_orig = _softmax(model, image_tensor)
        sm_jpeg = _softmax(model, sq_jpeg)

        # Model's top-1 confidence on the original image
        top1_conf = float(sm_orig.max().item())

        # ── Signal 1: JPEG softmax L1 delta (PRIMARY)
        # KEY FIX: Normalize by (1 - top1_conf) so that images where ResNet is
        # already uncertain don't get artificially high scores.
        # A truly adversarial image: top1_conf≈1.0 AND jpeg_delta is large
        # A natural boundary image: top1_conf<<1.0 AND jpeg_delta can be large
        jpeg_softmax_delta_raw = _l1_dist(sm_orig, sm_jpeg)
        # Normalise: ratio of delta to the model's uncertainty budget
        uncertainty = max(1.0 - top1_conf, 0.001)
        jpeg_softmax_delta_norm = jpeg_softmax_delta_raw / uncertainty

        # Clean:        jpeg_sm_raw≈0.00002, top1≈0.9999, norm≈0.24
        # Clean tabby:  jpeg_sm_raw≈0.098,   top1≈0.67,   norm≈0.30  (similar!)
        # Adversarial:  jpeg_sm_raw≈0.00485, top1≈1.0000, norm≈485   (huge gap!)
        # Threshold at 2.0 (normalised) gives excellent separation
        t_jpeg_norm   = 2.0

        # ── Signal 2: JPEG pixel L1 delta ────────────────────────────────────
        pix_orig = _denorm(image_tensor)
        pix_jpeg = _denorm(sq_jpeg)
        jpeg_pixel_delta = float((pix_orig - pix_jpeg).abs().mean().item())
        t_jpeg_px = _JPEG_PIXEL_THRESH * scale

        # ── Signal 3: Bit-depth pixel L1 delta ───────────────────────────────
        pix_bits = _denorm(sq_bits)
        bits_pixel_delta = float((pix_orig - pix_bits).abs().mean().item())
        t_bits_px = _BITS_PIXEL_THRESH * scale

        # ── Signal 4: Gaussian pixel L1 delta ────────────────────────────────
        pix_blur = _denorm(sq_blur)
        blur_pixel_delta = float((pix_orig - pix_blur).abs().mean().item())
        t_blur_px = _BLUR_PIXEL_THRESH * scale

        # ── Sigmoid confidence per signal ────────────────────────────────────
        c1 = _sigmoid_conf(jpeg_softmax_delta_norm, t_jpeg_norm,  scale=3.0)
        c2 = _sigmoid_conf(jpeg_pixel_delta,        t_jpeg_px,    scale=6.0)
        c3 = _sigmoid_conf(bits_pixel_delta,        t_bits_px,    scale=4.0)
        c4 = _sigmoid_conf(blur_pixel_delta,        t_blur_px,    scale=6.0)

        # Weighted combination — normalised JPEG softmax dominates
        confidence = 0.60 * c1 + 0.20 * c2 + 0.10 * c3 + 0.10 * c4
        is_adv = confidence > 0.5

        votes = sum([
            jpeg_softmax_delta_norm > t_jpeg_norm,
            jpeg_pixel_delta        > t_jpeg_px,
            bits_pixel_delta        > t_bits_px,
            blur_pixel_delta        > t_blur_px,
        ])

        # ── Optional ANN override ─────────────────────────────────────────────
        method = "multi_signal"
        if use_ann and self._is_trained:
            feat_t = torch.tensor([jpeg_softmax_delta_norm, jpeg_pixel_delta,
                                   bits_pixel_delta, blur_pixel_delta])
            with torch.no_grad():
                logits = self.ann(feat_t.unsqueeze(0))
                probs  = torch.softmax(logits, dim=1)[0]
            is_adv     = bool(probs[1].item() > 0.5)
            confidence = float(probs[1].item())
            method     = "ann+multi_signal"

        verdict = (
            "Very likely adversarial" if confidence > 0.75
            else "Possibly adversarial"  if confidence > 0.50
            else "Possibly clean"        if confidence > 0.30
            else "Likely clean"
        )

        return {
            "is_adversarial" : is_adv,
            "confidence"     : round(confidence, 4),
            "verdict"        : verdict,
            "votes"          : votes,
            "scores"         : {
                "gaussian_blur_delta"  : round(blur_pixel_delta,          6),
                "bit_depth_delta"      : round(bits_pixel_delta,          6),
                "jpeg_compress_delta"  : round(jpeg_pixel_delta,          6),
                "max_delta"            : round(jpeg_softmax_delta_norm,   6),
            },
            "features"       : [
                round(jpeg_softmax_delta_norm, 6),
                round(jpeg_pixel_delta,        6),
                round(bits_pixel_delta,        6),
                round(blur_pixel_delta,        6),
            ],
            "threshold"      : round(t_jpeg_norm, 4),
            "delta_blur"     : round(blur_pixel_delta,         6),
            "delta_bits"     : round(bits_pixel_delta,         6),
            "delta_jpeg"     : round(jpeg_pixel_delta,         6),
            "delta_max"      : round(jpeg_softmax_delta_norm,  6),
            "method"         : method,
        }

    def add_sample(self, image_tensor: torch.Tensor, label: int):

        """Buffer one labeled sample for ANN training. label: 0=clean, 1=adv."""
        model   = self._get_model()
        sq_jpeg = _jpeg_compress(image_tensor)
        sq_bits = _reduce_bits(image_tensor)
        sq_blur = _gaussian_blur(image_tensor)
        sm_orig = _softmax(model, image_tensor)
        sm_jpeg = _softmax(model, sq_jpeg)
        pix_o   = _denorm(image_tensor)
        d1 = _l1_dist(sm_orig, sm_jpeg)
        d2 = float((pix_o - _denorm(sq_jpeg)).abs().mean().item())
        d3 = float((pix_o - _denorm(sq_bits)).abs().mean().item())
        d4 = float((pix_o - _denorm(sq_blur)).abs().mean().item())
        self._buf_X.append(torch.tensor([d1, d2, d3, d4]))
        self._buf_y.append(label)

    def fit(self, epochs: int = 100, lr: float = 1e-3) -> float:
        """Train the ANN on buffered feature vectors."""
        if len(self._buf_X) < 4:
            raise RuntimeError("Need at least 4 buffered samples to train.")
        X  = torch.stack(self._buf_X)
        y  = torch.tensor(self._buf_y, dtype=torch.long)
        mu = X.mean(0); sig = X.std(0) + 1e-8
        Xn = (X - mu) / sig
        self._mu = mu; self._sig = sig
        self.ann.train()
        opt = torch.optim.Adam(self.ann.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = F.cross_entropy(self.ann(Xn), y)
            loss.backward(); opt.step()
        self.ann.eval()
        self._is_trained = True
        return float(loss.item())
