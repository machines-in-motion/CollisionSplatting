"""Download helpers for public example scenes.

The Marble sample worlds are hosted by World Labs (https://docs.worldlabs.ai/marble/export/specs); they are
fetched on demand from their CDN and are not redistributed with this package.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Optional

MARBLE_SAMPLES = (
    "rustic_kitchen_with_natural_light",
    "elegant_library_with_fireplace",
    "modern_house_with_lush_landscaping",
    "narrow_european_cobblestone_lane",
    "warm_traditional_kitchen_interior",
)
_MARBLE_URL = "https://wlt-ai-cdn.art/example_exports/{name}/{name}_{kind}"


def cache_dir() -> Path:
    """Download cache (``$COLLISIONSPLATTING_CACHE`` or ``~/.cache/collisionsplatting``)."""
    d = Path(os.environ.get("COLLISIONSPLATTING_CACHE", Path.home() / ".cache" / "collisionsplatting"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_marble_sample(name: str = "rustic_kitchen_with_natural_light", resolution: str = "500k",
                           collider: bool = False, directory: Optional[os.PathLike] = None) -> Path:
    """Download one of World Labs' public Marble sample worlds (a standard 3DGS ``.ply``) and return its path.

    Marble exports are in the OpenCV convention (x right, **y down**, z forward), roughly metric.

    Args:
        name: one of :data:`MARBLE_SAMPLES`.
        resolution: ``"500k"`` or ``"2m"`` splats.
        collider: download the coarse collider mesh (``.glb``) instead of the splats.
        directory: target directory (default :func:`cache_dir`).
    """
    if name not in MARBLE_SAMPLES:
        raise ValueError(f"unknown sample {name!r}; choose from {MARBLE_SAMPLES}")
    kind = "collider.glb" if collider else f"{resolution}.ply"
    out = Path(directory or cache_dir()) / f"{name}_{kind}"
    if not out.exists():
        url = _MARBLE_URL.format(name=name, kind=kind)
        print(f"downloading {url}")
        tmp = out.with_suffix(out.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(out)
    return out
