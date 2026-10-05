"""Full-page English manga OCR: detect -> recognize -> CTC/lexicon gate -> reading order.

Standalone extraction of the `manga-translator` project (its Phases 0-5),
trimmed to the English/Latin path: image in, `TextRegion` list out. The
inference recipe (crop geometry, batching, beam search, gate thresholds) is
kept identical to the origin project — `recognize.py` and `detect.py` hold the
model wrappers, and the numbers in DEFAULT_OPTIONS mirror the origin's
`configs/baseline.yaml` plus the gate chosen in exp-104/105/117.

Usage:
    from manga_page_ocr import PageOCR, load_image_rgb

    ocr = PageOCR()                       # models/ next to the package
    regions = ocr.run_sync(load_image_rgb('page.jpg'), 'page.jpg')
    for r in regions:
        print(r.id, r.bbox, r.text, r.confidence, r.flag)

    # inside an async app, load once and reuse:
    await ocr.load()
    regions = await ocr.run(image_rgb, 'page.jpg')
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from manga_page_ocr.detect import DBNetDetector
from manga_page_ocr.lexicon import reads_more_like_english
from manga_page_ocr.reading_order import sort_regions
from manga_page_ocr.recognize import Mit48pxCTCOCR, Mit48pxOCR
from manga_page_ocr.schema import TextRegion
from manga_page_ocr.validity import check_text, clean_control_tokens

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PACKAGE_DIR.parent / 'models'

# Reference values from the origin project's configs/baseline.yaml.
DEFAULT_OPTIONS: dict = {
    'device': 'cuda',
    'detector': {
        'detect_size': 2048,
        'text_threshold': 0.5,
        'box_threshold': 0.7,
        'unclip_ratio': 2.3,
    },
    'ocr': {
        'text_height': 32,
        'beams_k': 5,          # exp-109 lowered it, exp-110 reverted: user pages regressed
        'max_seq_length': 255,
        'min_prob': 0.7,
        'padding_pct': 5.0,    # exp-002 (adopted)
        'src_scale': 1.0,      # exp-003 (rejected: no gain)
        'binarize': 'none',    # exp-004 (rejected)
    },
    'fallback': {
        'enabled': True,
        'gate_min_prob': 0.75,  # plan Phase 3 accept threshold
        'disagreement': True,   # exp-105 (adopted)
        'ctc_min_prob': 0.5,
    },
    'reading_order': {
        'right_to_left': False,  # English pages; set True for Japanese-style pages
    },
}


def _merged(base: dict, overrides: Optional[dict]) -> dict:
    """One-level-deep merge of nested option dicts."""
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in base.items()}
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def load_image_rgb(path: str | Path) -> np.ndarray:
    return np.array(Image.open(path).convert('RGB'))


def quad_to_region(image_name: str, idx: int, q, detector: str, ocr: str,
                   language: str = 'en') -> TextRegion:
    """Convert a recognized Quadrilateral into the plan's schema.

    Copied verbatim from the origin project's src/pipeline.py.
    """
    xs, ys = q.pts[:, 0], q.pts[:, 1]
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    region = TextRegion(
        image=image_name,
        id=idx,
        bbox=[x1, y1, x2, y2],
        polygon=[[int(x), int(y)] for x, y in q.pts],
        text=q.text if q.text is not None else '',
        confidence=float(q.prob),
        direction='vertical' if q.direction == 'v' else 'horizontal',
        language=language,
        detection_score=float(q.prob_det) if hasattr(q, 'prob_det') else 0.0,
        fg_color=[q.fg_r, q.fg_g, q.fg_b] if hasattr(q, 'fg_r') else None,
        bg_color=[q.bg_r, q.bg_g, q.bg_b] if hasattr(q, 'bg_r') else None,
        flag=getattr(q, 'flag', None) or ('low_confidence' if getattr(q, 'flagged', False) else None),
        detector=detector,
        ocr=getattr(q, 'ocr_chain', ocr),
        primary_text=getattr(q, 'primary_text', None),
        primary_confidence=getattr(q, 'primary_prob', None),
    )
    return region


def save_jsonl(regions: list[TextRegion], path: str | Path) -> None:
    """One TextRegion per line (origin project's results format)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fp:
        for r in regions:
            fp.write(json.dumps(r.to_dict(), ensure_ascii=False) + '\n')


async def apply_quality_gate(image: np.ndarray, quads: list, primary: Mit48pxOCR,
                             fallback: Mit48pxCTCOCR, cfg: dict) -> None:
    """Phase-3 quality gate (origin project's src/pipeline.py, copied verbatim).

    1. primary conf >= gate AND text valid -> accept
    2. else -> OCR-CTC; accept if good, else keep the more-confident backend
       + flag 'needs_review'
    3. (exp-105) disagreement check: run both backends on the confident boxes
       too, and let the exp-117 lexicon decide which reading is English.
    """
    fb = cfg.get('fallback') or {}
    gate = fb.get('gate_min_prob', 0.75)
    ctc_min = fb.get('ctc_min_prob', 0.5)
    if not fb.get('enabled', True) or fallback is None:
        for q in quads:
            if q.prob < gate:
                q.flag = 'low_confidence'
        return
    language = cfg.get('language', 'en')
    low = []
    for q in quads:
        ok, _ = check_text(q.text or '', language)
        if q.prob >= gate and ok:
            q.flag = None
        else:
            low.append(q)
    if low:
        logging.info('fallback: %d box(es) below gate -> OCR-CTC', len(low))
        await fallback.recognize(image, low)
        for q in low:
            ok, _ = check_text(q.text or '', language)
            if q.prob >= ctc_min and ok:
                q.flag = None
                q.ocr_chain = fallback.name
            else:
                # keep whichever backend was more confident, flag for review
                if q.primary_prob >= q.prob:
                    q.text, q.prob = q.primary_text, q.primary_prob
                q.flag = 'needs_review'
                q.ocr_chain = f'{primary.name}+{fallback.name}'
    if fb.get('disagreement', False):
        others = [q for q in quads if q not in low]
        if others:
            await fallback.recognize(image, others)
            for q in others:
                ctc_text, ctc_prob = q.text, q.prob
                # restore primary as the accepted baseline
                q.text, q.prob = q.primary_text, q.primary_prob
                # exp-117: the lexicon, not the confidence, decides which
                # reading is English (see lexicon.py for the failure modes)
                if reads_more_like_english(q.primary_text or '', ctc_text or ''):
                    logging.info('lexicon: %r -> %r', q.primary_text, ctc_text)
                    q.text, q.prob = ctc_text, ctc_prob
                    q.ocr_chain = fallback.name


class PageOCR:
    """Holds the loaded models so a server can reuse them across pages."""

    def __init__(self, model_dir: str | Path | None = None, device: str | None = None,
                 right_to_left: bool = False, options: Optional[dict] = None):
        self.options = _merged(DEFAULT_OPTIONS, options)
        if device is not None:
            self.options['device'] = device
        if right_to_left:
            self.options['reading_order']['right_to_left'] = True
        self.device = self.options['device']
        self.model_dir = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR

        opts = self.options
        self.detector = DBNetDetector(self.model_dir, device=self.device)
        self.ocr = Mit48pxOCR(
            self.model_dir, device=self.device,
            text_height=opts['ocr']['text_height'],
            beams_k=opts['ocr']['beams_k'],
            max_seq_length=opts['ocr']['max_seq_length'],
            min_prob=opts['ocr']['min_prob'],
            padding_pct=opts['ocr']['padding_pct'],
            src_scale=opts['ocr']['src_scale'],
            binarize=opts['ocr']['binarize'],
        )
        # CTC always consumes raw crops (padding_pct left at its 0.0 default) —
        # matches the origin project's make_fallback_ocr().
        self.fallback = Mit48pxCTCOCR(
            self.model_dir, device=self.device,
            min_prob=opts['fallback'].get('ctc_min_prob', 0.5),
        ) if opts['fallback'].get('enabled', True) else None
        self._loaded = False

    async def load(self) -> None:
        logging.info('loading models from %s on %s...', self.model_dir, self.device)
        await self.detector.load()
        await self.ocr.load()
        if self.fallback is not None:
            await self.fallback.load()
        self._loaded = True
        logging.info('models loaded')

    async def run(self, image: np.ndarray, image_name: str = 'page') -> list[TextRegion]:
        """OCR one RGB page (numpy HxWx3). Returns regions in reading order."""
        if not self._loaded:
            await self.load()
        cfg = self.options
        quads, _raw_mask = (await self.detector.detect(image, **cfg['detector']))[:2]
        # keep detection score before OCR overwrites prob
        for q in quads:
            q.prob_det = q.prob

        await self.ocr.recognize(image, quads)
        for q in quads:
            q.primary_text, q.primary_prob = q.text, q.prob
            q.ocr_chain = self.ocr.name

        await apply_quality_gate(image, quads, self.ocr, self.fallback, cfg)

        # '<UNK>' is the recognizer saying it cannot read the crop at all — a
        # control token, not text. Emptying it flags the box for review instead.
        for q in quads:
            q.text, unreadable = clean_control_tokens(q.text or '')
            if unreadable:
                q.flag = 'needs_review'

        quads = sorted(quads, key=lambda q: (q.pts[:, 1].min(), q.pts[:, 0].min()))
        regions = [quad_to_region(image_name, i, q, self.detector.name, self.ocr.name, 'en')
                   for i, q in enumerate(quads)]

        # Phase 4: panel-aware reading order (fallback: simple top-to-bottom)
        rtl = cfg['reading_order'].get('right_to_left', False)
        regions, _panels = sort_regions(regions, image, right_to_left=rtl)
        for i, r in enumerate(regions):
            r.id = i
        return regions

    def run_sync(self, image: np.ndarray, image_name: str = 'page') -> list[TextRegion]:
        """Blocking convenience wrapper (fails inside a running event loop —
        use `await run(...)` there instead)."""
        return asyncio.run(self.run(image, image_name))
