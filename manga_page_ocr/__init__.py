"""Standalone full-page English manga OCR (extracted from `manga-translator`).

    from manga_page_ocr import PageOCR, load_image_rgb, save_jsonl

    ocr = PageOCR()                                   # models/ next to the package
    regions = ocr.run_sync(load_image_rgb('page.jpg'), 'page.jpg')
    save_jsonl(regions, 'page.jsonl')

See page.py for the option knobs and README.md for provenance + licence notes.
"""
from manga_page_ocr.page import (
    DEFAULT_MODEL_DIR,
    DEFAULT_OPTIONS,
    PageOCR,
    apply_quality_gate,
    load_image_rgb,
    quad_to_region,
    save_jsonl,
)
from manga_page_ocr.schema import TextRegion

__all__ = [
    'DEFAULT_MODEL_DIR',
    'DEFAULT_OPTIONS',
    'PageOCR',
    'TextRegion',
    'apply_quality_gate',
    'load_image_rgb',
    'quad_to_region',
    'save_jsonl',
]
