"""Result schema per "Manga OCR — Loop Engineering Plan" (minimal schema + extras)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class TextRegion:
    """One detected + recognized text region.

    Required by the plan:
      image, id, bbox [x1,y1,x2,y2], polygon (4 points), text,
      confidence, direction ('horizontal'|'vertical'), language

    Extras (kept for eval/Phase-6 work, cheap to store):
      detection_score, fg_color, bg_color, flag, detector, ocr
    """
    image: str
    id: int
    bbox: List[int]                 # [x1, y1, x2, y2] axis-aligned
    polygon: List[List[int]]        # 4 points
    text: str
    confidence: float               # OCR probability (mean beam logprob exp)
    direction: str                  # 'horizontal' | 'vertical'
    language: str = 'en'
    panel_index: Optional[int] = None  # Phase 4: reading-order panel id
    detection_score: float = 0.0    # detector score for the box
    fg_color: Optional[List[int]] = None   # [r, g, b] predicted text color
    bg_color: Optional[List[int]] = None   # [r, g, b] predicted bg color
    flag: Optional[str] = None      # 'low_confidence' | 'needs_review'
    detector: str = ''
    ocr: str = ''                   # backend(s) that produced the final text
    primary_text: Optional[str] = None        # primary OCR output (pre-fallback)
    primary_confidence: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)
