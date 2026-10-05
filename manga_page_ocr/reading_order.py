"""Phase 4 — panel-aware reading order (plan: panel index -> bubble centroid ->
language direction). Adapted from zyddnys/manga-image-translator utils/sort.py
(sort_regions + _sort_panels_fill), working on our TextRegion schema.

Ordering policy:
  1. detect panels with Kumiko (vendored) in reading order (rtl flag)
  2. assign each region to its panel (center point; nearest-panel fallback)
  3. sort regions per panel: if they spread more horizontally, read by x
     (rtl-aware) grouping rows; otherwise by y grouping columns
  4. fall back to the simple top-to-bottom sort if panel detection fails
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np

from manga_page_ocr.schema import TextRegion
from manga_page_ocr.vendor.utils.panel import get_panels_from_array


def _region_center(r: TextRegion) -> Tuple[float, float]:
    x1, y1, x2, y2 = r.bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _simple_sort(regions: List[TextRegion], right_to_left: bool) -> List[TextRegion]:
    """Fallback: top-to-bottom, then x in reading direction."""
    sorted_regions: List[TextRegion] = []
    for r in sorted(regions, key=lambda r: _region_center(r)[1]):
        cx, cy = _region_center(r)
        for i, s in enumerate(sorted_regions):
            sx1, sy1, sx2, sy2 = s.bbox
            if cy > sy2:
                continue
            if cy < sy1:
                sorted_regions.insert(i, r)
                break
            scx = (sx1 + sx2) / 2
            if (right_to_left and cx > scx) or (not right_to_left and cx < scx):
                sorted_regions.insert(i, r)
                break
        else:
            sorted_regions.append(r)
    return sorted_regions


def _sort_panels_fill(panels: List[Tuple[int, int, int, int]], right_to_left: bool) -> List[Tuple[int, int, int, int]]:
    """Panels in reading order: rows top-to-bottom; within a row in reading
    direction; same-column stacks kept together (reference logic)."""
    if not panels:
        return panels
    remaining = sorted(list(panels), key=lambda p: p[1])
    ordered: List[Tuple[int, int, int, int]] = []
    avg_w = np.mean([p[2] - p[0] for p in remaining])
    avg_h = np.mean([p[3] - p[1] for p in remaining])
    x_thr = max(10, avg_w * 0.1)
    y_thr = max(10, avg_h * 0.3)
    while remaining:
        base_y = remaining[0][1]
        row = []
        i = 0
        while i < len(remaining):
            if abs(remaining[i][1] - base_y) <= y_thr:
                row.append(remaining.pop(i))
            else:
                i += 1
        row.sort(key=lambda p: (-p[0] if right_to_left else p[0]))
        ordered.extend(row)
    return ordered


def sort_regions(regions: List[TextRegion], image: np.ndarray,
                 right_to_left: bool = False,
                 ) -> Tuple[List[TextRegion], List[Tuple[int, int, int, int]]]:
    """Returns (regions in reading order, panels) and sets panel_index per region."""
    if not regions:
        return [], []
    panels: List[Tuple[int, int, int, int]] = []
    try:
        raw = get_panels_from_array(image, rtl=right_to_left)
        panels = [(x, y, x + w, y + h) for x, y, w, h in raw]
        panels = _sort_panels_fill(panels, right_to_left)
    except Exception as e:
        logging.warning('panel detection failed (%s), using simple sort', e)
        return _simple_sort(regions, right_to_left), panels

    for r in regions:
        cx, cy = _region_center(r)
        r.panel_index = -1
        for idx, (x1, y1, x2, y2) in enumerate(panels):
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                r.panel_index = idx
                break
        if r.panel_index < 0:
            dists = [((max(x1 - cx, 0, cx - x2)) ** 2 + (max(y1 - cy, 0, cy - y2)) ** 2, i)
                     for i, (x1, y1, x2, y2) in enumerate(panels)]
            if dists:
                r.panel_index = min(dists)[1]

    grouped = {}
    for r in regions:
        grouped.setdefault(r.panel_index, []).append(r)

    sorted_all: List[TextRegion] = []
    for pi in sorted(grouped.keys()):
        group = grouped[pi]
        xs = [c[0] for c in map(_region_center, group)]
        ys = [c[1] for c in map(_region_center, group)]
        x_std = np.std(xs) if len(xs) > 1 else 0
        y_std = np.std(ys) if len(ys) > 1 else 0
        if x_std > y_std:
            primary = sorted(group, key=lambda r: -_region_center(r)[0] if right_to_left else _region_center(r)[0])
            # reference groups x-columns within 20px, then sorts by y — replicate:
            out, sub, prev = [], [], None
            for r in primary:
                cx = _region_center(r)[0]
                if prev is not None and abs(cx - prev) > 20:
                    sub.sort(key=lambda r: _region_center(r)[1])
                    out += sub
                    sub = []
                sub.append(r)
                prev = cx
            if sub:
                sub.sort(key=lambda r: _region_center(r)[1])
                out += sub
            sorted_all += out
        else:
            primary = sorted(group, key=lambda r: _region_center(r)[1])
            out, sub, prev = [], [], None
            for r in primary:
                cy = _region_center(r)[1]
                if prev is not None and abs(cy - prev) > 15:
                    sub.sort(key=lambda r: -_region_center(r)[0] if right_to_left else _region_center(r)[0])
                    out += sub
                    sub = []
                sub.append(r)
                prev = cy
            if sub:
                sub.sort(key=lambda r: -_region_center(r)[0] if right_to_left else _region_center(r)[0])
                out += sub
            sorted_all += out
    return sorted_all, panels


def draw_panels(image: np.ndarray, panels: List[Tuple[int, int, int, int]]) -> np.ndarray:
    """Magenta panel boxes with order numbers (for the user to verify)."""
    img = image.copy()
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)
    for idx, (x1, y1, x2, y2) in enumerate(panels):
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), lw)
        cv2.putText(img, str(idx), (x1 + 5, y1 + 60), cv2.FONT_HERSHEY_SIMPLEX,
                    lw / 2, (200, 100, 0), max(lw - 1, 1), cv2.LINE_AA)
    return img
