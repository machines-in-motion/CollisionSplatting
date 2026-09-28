"""Small helpers shared by the examples (not part of the library API)."""
import os
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("CS_OUTPUT_DIR", Path(__file__).resolve().parent / "outputs"))
OUT.mkdir(parents=True, exist_ok=True)


def save_png(img, name):
    """Save an (H, W, 3) float [0, 1] or uint8 RGB image under ``examples/outputs``."""
    import imageio.v3 as iio
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    path = OUT / name
    iio.imwrite(path, img)
    print(f"saved {path}")
    return path
