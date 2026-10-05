"""Text detector loader — DBNet (`detect-20241225.ckpt`) from manga-image-translator.

The model code is vendored in `manga_page_ocr/vendor/detection` (slimmed from
zyddnys/manga-image-translator, GPL-3.0 — see the bundle README).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from manga_page_ocr.vendor.detection import DefaultDetector
from manga_page_ocr.vendor.utils import Quadrilateral


class DBNetDetector:
    """Wraps the reference DefaultDetector, pinning the model dir from config."""

    name = 'dbnet'

    def __init__(self, model_dir: str | Path, device: str = 'cuda'):
        # ModelWrapper resolves files as <_MODEL_DIR>/<_MODEL_SUB_DIR>/<name>;
        # our downloads live directly in model_dir.
        DefaultDetector._MODEL_DIR = str(Path(model_dir).resolve())
        DefaultDetector._MODEL_SUB_DIR = ''
        self._detector = DefaultDetector()
        self.device = device

    async def load(self) -> None:
        await self._detector.load(self.device)

    async def detect(self, image: np.ndarray, detect_size: int, text_threshold: float,
                     box_threshold: float, unclip_ratio: float,
                     ) -> Tuple[List[Quadrilateral], np.ndarray]:
        """Returns (textlines, raw_text_mask). Image is RGB numpy."""
        return await self._detector.detect(
            image, detect_size, text_threshold, box_threshold, unclip_ratio,
            invert=False, gamma_correct=False, rotate=False, auto_rotate=False,
            verbose=False,
        )
