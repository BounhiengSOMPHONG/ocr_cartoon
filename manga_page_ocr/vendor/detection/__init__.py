"""Slimmed re-exports from zyddnys/manga-image-translator detection.

Only the DBNet default detector is exposed (ctd, dbnet_convnext, craft,
paddle_rust are intentionally not imported to keep the dependency surface
small). ComicTextDetector was dropped together with comictextdetector.pt in
the origin project's 2026-10-05 cleanup and is not part of this bundle.
"""
from .common import CommonDetector, OfflineDetector
from .default import DefaultDetector
