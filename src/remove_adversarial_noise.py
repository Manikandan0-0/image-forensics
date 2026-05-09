"""
Module implementing adversarial noise REMOVAL strategies.

Each method works in the normalized domain [batch, 3, H, W] (ImageNet stats).
The pipeline for every method:
  1. Denormalize → [0, 1] pixel space
  2. Apply denoising
  3. Re-normalize back to ImageNet stats
  4. Return tensor of the same shape as the input
"""

import io
import torch
import torchvision.transforms as T
import kornia
from PIL import Image

from utils import denormalize, MEAN, STANDARD_DEVIATION


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _to_pixel(x: torch.Tensor, mean, std) -> torch.Tensor:
    """Denormalize from ImageNet stats to [0, 1] pixel space."""
    return denormalize(x, mean, std).clamp(0.0, 1.0)


def _to_norm(x: torch.Tensor, mean, std) -> torch.Tensor:
    """Re-normalize [0, 1] pixel space back to ImageNet stats."""
    return (x - mean) / std


def compute_psnr(original: torch.Tensor, denoised: torch.Tensor,
                 mean, std) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two normalized tensors.
    Both tensors are first denormalized to [0, 1] before computing PSNR.
    Higher PSNR → denoised image is closer to the reference.
    """
    orig_pix = _to_pixel(original.detach(), mean, std)
    den_pix  = _to_pixel(denoised.detach(), mean, std)
    # kornia.metrics.psnr expects values in [0, max_val]
    psnr_val = kornia.metrics.psnr(den_pix, orig_pix, max_val=1.0)
    return float(psnr_val.item())


def compute_ssim(original: torch.Tensor, denoised: torch.Tensor,
                 mean, std) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two normalized tensors.
    Both tensors are denormalized to [0, 1] before computing SSIM.
    SSIM ∈ [0, 1] — values close to 1 mean near-identical structure.
    """
    orig_pix = _to_pixel(original.detach(), mean, std)
    den_pix  = _to_pixel(denoised.detach(), mean, std)
    # kornia.metrics.ssim returns a tensor map; take the mean over spatial dims
    ssim_map = kornia.metrics.ssim(den_pix, orig_pix, window_size=11)
    return float(ssim_map.mean().item())


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AdversarialDenoiser:
    """
    Implements multiple denoising defense strategies against adversarial
    perturbations.

    All public methods accept / return tensors in *normalized* ImageNet space:
        shape  : [1, 3, H, W]
        dtype  : float32
        values : roughly in [-2.5, 2.5] (after ImageNet normalization)
    """

    def __init__(self, epsilon: float = 0.05):
        """
        Args:
            epsilon: The L-∞ bound used during the attack (informational —
                     used to size certain defense parameters adaptively).
        """
        self.epsilon = epsilon
        # Pre-create mean/std tensors shaped for broadcast [1, 3, 1, 1]
        self.mean = torch.tensor(MEAN).view(1, 3, 1, 1)
        self.std  = torch.tensor(STANDARD_DEVIATION).view(1, 3, 1, 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def denoise(self, image_tensor: torch.Tensor, method: str = "tv",
                **kwargs) -> torch.Tensor:
        """
        Dispatch to the selected denoising strategy.

        Args:
            image_tensor: Normalized input tensor [1, 3, H, W].
            method      : One of {"gaussian", "tv", "jpeg",
                          "feature_squeezing", "randomized_smoothing"}.
            **kwargs    : Extra hyper-parameters forwarded to the strategy.

        Returns:
            Denoised normalized tensor of the same shape.
        """
        method = method.lower().replace(" ", "_").replace("-", "_")
        dispatch = {
            "gaussian"             : self._gaussian,
            "tv"                   : self._tv_denoise,
            "jpeg"                 : self._jpeg_compression,
            "feature_squeezing"    : self._feature_squeezing,
            "randomized_smoothing" : self._randomized_smoothing,
        }
        if method not in dispatch:
            raise ValueError(
                f"Unknown denoising method '{method}'. "
                f"Choose from: {list(dispatch.keys())}"
            )
        return dispatch[method](image_tensor, **kwargs)

    def compare_methods(self, image_tensor: torch.Tensor) -> dict:
        """
        Run all available denoising strategies on the same input.

        Returns:
            Dict mapping method name → denoised tensor.
        """
        methods = [
            "gaussian", "tv", "jpeg",
            "feature_squeezing", "randomized_smoothing",
        ]
        return {m: self.denoise(image_tensor, method=m) for m in methods}

    # ------------------------------------------------------------------
    # Strategy Implementations
    # ------------------------------------------------------------------

    def _gaussian(self, x: torch.Tensor,
                  kernel_size: int = 5,
                  sigma: float = 1.0) -> torch.Tensor:
        """
        Gaussian Smoothing Defense.

        Theory: Adversarial perturbations tend to be high-frequency signals.
        A Gaussian blur low-pass filters the image and blurs away the sharp,
        structured noise while keeping the coarse visual content intact.

        Args:
            kernel_size: Odd integer for the blur kernel size.
            sigma      : Standard deviation of the Gaussian kernel.
        """
        # Operate directly in normalized space — linear filter commutes with
        # the affine normalization, so results are equivalent.
        blur = T.GaussianBlur(kernel_size=kernel_size, sigma=(sigma, sigma))
        return blur(x)

    def _tv_denoise(self, x: torch.Tensor,
                    weight: float = 0.1,
                    iters: int = 50) -> torch.Tensor:
        """
        Total Variation (TV) Denoising.

        Theory: TV denoising solves the optimization:
            min_z  0.5 * ||z - x||² + λ * TV(z)
        where TV(z) = Σ |∇z| penalizes rapid spatial changes (noise) while
        the data-fidelity term keeps z close to the input.  This removes
        structured adversarial perturbations while preserving object edges.

        Args:
            weight: Regularization strength λ.  Higher → smoother result.
            iters : Number of gradient-descent iterations.
        """
        # Clone and enable gradients on the denoised variable
        denoised = x.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([denoised], lr=0.01)

        for _ in range(iters):
            optimizer.zero_grad()
            # Data-fidelity: stay close to the (adversarial) input
            mse_loss = torch.nn.functional.mse_loss(denoised, x.detach())
            # Regularizer: penalize high-frequency spatial variation
            tv_loss  = kornia.losses.total_variation(denoised).mean()
            loss = mse_loss + weight * tv_loss
            loss.backward()
            optimizer.step()

        return denoised.detach()

    def _jpeg_compression(self, x: torch.Tensor,
                           quality: int = 75) -> torch.Tensor:
        """
        JPEG Compression Defense.

        Theory: JPEG's block-DCT quantization naturally discards high-frequency
        components of an image, which is exactly where adversarial noise lives.
        A round-trip through JPEG encoding/decoding at moderate quality (e.g.
        75) destroys most ε-bounded perturbations while keeping perceptual
        quality acceptable.

        Args:
            quality: JPEG quality factor [1-95].  Lower → more compression
                     (more noise removed, but more image distortion).
        """
        # 1. Denormalize to [0, 1] pixel space
        x_pix = _to_pixel(x, self.mean, self.std).squeeze(0)  # [3, H, W]

        # 2. Convert to PIL image and save as JPEG into an in-memory buffer
        img_pil = T.ToPILImage()(x_pix)
        buf = io.BytesIO()
        img_pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)

        # 3. Reload the JPEG-compressed image
        img_jpeg = Image.open(buf).convert("RGB")
        x_jpeg   = T.ToTensor()(img_jpeg).unsqueeze(0)  # [1, 3, H, W]

        # 4. Re-normalize to ImageNet stats
        return _to_norm(x_jpeg, self.mean, self.std)

    def _feature_squeezing(self, x: torch.Tensor,
                            bit_depth: int = 4) -> torch.Tensor:
        """
        Feature Squeezing Defense.

        Theory: Adversarial perturbations are typically small (ε ≈ 0.05 in
        pixel space ≈ ~13 out of 255 gray levels).  By reducing the effective
        bit depth of each pixel channel (e.g. from 8-bit to 4-bit), we
        quantize pixel values to a coarse grid, rounding away perturbations
        that are smaller than one quantization step (≈ 1/2^depth ≈ 0.063 for
        4-bit), effectively squeezing out the adversarial signal.

        Args:
            bit_depth: Target bit depth per channel.  4 → 16 discrete levels.
        """
        # 1. Denormalize to [0, 1]
        x_pix = _to_pixel(x, self.mean, self.std)

        # 2. Quantize: round to the nearest value on the reduced grid
        max_val    = (2 ** bit_depth) - 1          # e.g. 15 for 4-bit
        x_squeezed = torch.round(x_pix * max_val) / max_val

        # 3. Re-normalize
        return _to_norm(x_squeezed, self.mean, self.std)

    def _randomized_smoothing(self, x: torch.Tensor,
                               noise_std: float = 0.05,
                               n_samples: int = 20) -> torch.Tensor:
        """
        Randomized Smoothing Defense.

        Theory: Randomized smoothing certifiably defends against L₂-bounded
        attacks by averaging model-level (or input-level) outputs over many
        noisy copies of the input.  At the *input* level, averaging n noisy
        samples suppresses the structured adversarial signal (which has a
        fixed direction) relative to the i.i.d. random noise (which averages
        to zero), effectively recovering a cleaner approximation of the
        original image.

        Args:
            noise_std: Standard deviation of injected Gaussian noise (in
                       normalized space).  Should be ≥ epsilon for best effect.
            n_samples : Number of noisy copies to average.
        """
        # Generate n_samples noisy versions and average them
        # Shape: [n_samples, 1, 3, H, W]  ← broadcast-friendly
        noise  = torch.randn(n_samples, *x.shape) * noise_std
        copies = x.unsqueeze(0) + noise   # [N, 1, 3, H, W]
        return copies.mean(dim=0)         # [1, 3, H, W]
