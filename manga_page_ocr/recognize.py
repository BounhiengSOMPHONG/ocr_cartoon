"""OCR recognizer loader — MIT48px (`ocr.ckpt` + `alphabet-all-v5.txt`).

Model classes are vendored in `manga_page_ocr/vendor/ocr/mit48px.py` (pure-torch part of
zyddnys/manga-image-translator @ da668d7^, the last revision matching this
checkpoint). This wrapper replicates the reference inference recipe exactly:

  - crop each detected quad with `get_transformed_region` at `text_height` px
    (32 in the reference), rotating vertical text to horizontal
  - batch 16 crops sorted by width, zero-padded to the batch's max width
  - beam search: beams_k=5, max_seq_length=255
  - tokens: <S> skipped, </S> stops, <SP> -> space
  - reference dropped boxes with prob < 0.7; we keep them flagged
    `low_confidence` instead so detection/OCR eval still sees the box
    (deviation recorded in experiments/exp-001).
"""
from __future__ import annotations

import torch
import cv2
import einops
import numpy as np
from pathlib import Path
from typing import List

from manga_page_ocr.vendor.utils import Quadrilateral, chunks
from manga_page_ocr.vendor.ocr.mit48px import OCR
from manga_page_ocr.vendor.ocr.mit48px_ctc import OCR as CTCOCR


def get_region(image: np.ndarray, q: Quadrilateral, direction: str, text_height: int,
               padding_pct: float = 0.0, src_scale: float = 1.0,
               binarize: str = 'none') -> np.ndarray:
    """Quadrilateral crop for OCR input.

    Replicates the reference `Quadrilateral.get_transformed_region` exactly
    (homography of the quad to `text_height` px, vertical text rotated), plus:
      - `padding_pct`: crop margin, % of quad size (exp-002)
      - `src_scale`: upscale the crop 2x/3x before warping (exp-003). The model
        input height is fixed at 32px by design (assert in the model code), so
        "resize" must happen on the source: the 32px warp then samples more
        source pixels (super-sampling).
      - `binarize`: 'otsu' | 'adaptive' — threshold the 32px crop before the
        model (exp-004). Manga screentone/halftone backgrounds corrupt glyphs;
        the MIT48px model was trained on clean synthetic text. Polarity is
        chosen per crop: the minority pixel class is assumed to be the text
        strokes (strokes always cover less area than background in a text line).
    With padding_pct=0, src_scale=1 and binarize='none' the result must be
    pixel-identical to the reference.
    """
    [l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in q.structure]
    v_vec = l1b - l1a
    h_vec = l2b - l2a
    ratio = np.linalg.norm(v_vec) / np.linalg.norm(h_vec)

    src_pts = q.pts.astype(np.int64).copy()
    im_h, im_w = image.shape[:2]

    x1, y1, x2, y2 = src_pts[:, 0].min(), src_pts[:, 1].min(), src_pts[:, 0].max(), src_pts[:, 1].max()
    pad_x = int(round((x2 - x1) * padding_pct / 100.0))
    pad_y = int(round((y2 - y1) * padding_pct / 100.0))
    x1 = np.clip(x1 - pad_x, 0, im_w)
    y1 = np.clip(y1 - pad_y, 0, im_h)
    x2 = np.clip(x2 + pad_x, 0, im_w)
    y2 = np.clip(y2 + pad_y, 0, im_h)
    # cv2.warpPerspective could overflow if image size is too large, better crop it here
    img_croped = image[y1: y2, x1: x2]

    src_pts[:, 0] -= x1
    src_pts[:, 1] -= y1

    if src_scale != 1.0:
        img_croped = cv2.resize(img_croped, None, fx=src_scale, fy=src_scale,
                                interpolation=cv2.INTER_CUBIC)
        src_pts = src_pts.astype(np.float32) * src_scale

    def _binarize(region: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        if binarize == 'otsu':
            _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif binarize == 'adaptive':
            bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 25, 10)
        else:
            return region
        dark_frac = (bin_img == 0).mean()
        if dark_frac > 0.5:  # light text on dark background
            bin_img = 255 - bin_img
        return cv2.cvtColor(bin_img, cv2.COLOR_GRAY2RGB)

    if direction == 'h':
        h = max(int(text_height), 2)
        w = max(int(round(text_height / ratio)), 2)
        dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        region = cv2.warpPerspective(img_croped, M, (w, h))
        return _binarize(region)
    elif direction == 'v':
        w = max(int(text_height), 2)
        h = max(int(round(text_height * ratio)), 2)
        dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        region = cv2.warpPerspective(img_croped, M, (w, h))
        region = cv2.rotate(region, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return _binarize(region)


class Mit48pxOCR:
    name = 'mit48px'

    def __init__(self, model_dir: str | Path, device: str = 'cuda',
                 text_height: int = 32, beams_k: int = 5,
                 max_seq_length: int = 255, min_prob: float = 0.7,
                 padding_pct: float = 0.0, src_scale: float = 1.0,
                 binarize: str = 'none'):
        self.model_dir = Path(model_dir)
        self.device = device
        self.text_height = text_height
        self.beams_k = beams_k
        self.max_seq_length = max_seq_length
        self.min_prob = min_prob
        self.padding_pct = padding_pct
        self.src_scale = src_scale
        self.binarize = binarize
        self.max_chunk_size = 16

    async def load(self) -> None:
        with open(self.model_dir / 'ocr' / 'alphabet-all-v5.txt', encoding='utf-8') as fp:
            self.dictionary = [s[:-1] for s in fp.readlines()]
        self.model = OCR(self.dictionary, 768)
        sd = torch.load(self.model_dir / 'ocr' / 'ocr.ckpt', map_location='cpu')
        self.model.load_state_dict(sd['model'] if 'model' in sd else sd)
        self.model.eval()
        if self.device.startswith('cuda'):
            self.model = self.model.cuda()

    async def recognize(self, image: np.ndarray, quadrilaterals: List[Quadrilateral],
                        ) -> List[Quadrilateral]:
        """Fills .text/.prob/.fg_*/.bg_* on each quad. Returns the list."""
        if not quadrilaterals:
            return quadrilaterals

        region_imgs = [get_region(image, q, q.direction, self.text_height,
                                  self.padding_pct, self.src_scale, self.binarize)
                       for q in quadrilaterals]
        # Reference order: process short crops first (better batch padding).
        perm = sorted(range(len(region_imgs)), key=lambda x: region_imgs[x].shape[1])

        for indices in chunks(perm, self.max_chunk_size):
            N = len(indices)
            widths = [region_imgs[i].shape[1] for i in indices]
            max_width = 4 * (max(widths) + 7) // 4
            region = np.zeros((N, self.text_height, max_width, 3), dtype=np.uint8)
            for i, idx in enumerate(indices):
                region[i, :, :widths[i], :] = region_imgs[idx]

            image_tensor = (torch.from_numpy(region).float() - 127.5) / 127.5
            image_tensor = einops.rearrange(image_tensor, 'N H W C -> N C H W')
            if self.device.startswith('cuda'):
                image_tensor = image_tensor.cuda()
            with torch.no_grad():
                ret = self.model.infer_beam_batch(
                    image_tensor, widths, beams_k=self.beams_k,
                    max_seq_length=self.max_seq_length)

            for i, (pred_chars_index, prob, fr, fg, fb, br, bg, bb) in enumerate(ret):
                q = quadrilaterals[indices[i]]
                q.prob = prob
                q.fg_r = int(torch.clip(fr.view(-1), 0, 1).mean() * 255)
                q.fg_g = int(torch.clip(fg.view(-1), 0, 1).mean() * 255)
                q.fg_b = int(torch.clip(fb.view(-1), 0, 1).mean() * 255)
                q.bg_r = int(torch.clip(br.view(-1), 0, 1).mean() * 255)
                q.bg_g = int(torch.clip(bg.view(-1), 0, 1).mean() * 255)
                q.bg_b = int(torch.clip(bb.view(-1), 0, 1).mean() * 255)
                seq = []
                for chid in pred_chars_index:
                    ch = self.dictionary[chid]
                    if ch == '<S>':
                        continue
                    if ch == '</S>':
                        break
                    if ch == '<SP>':
                        ch = ' '
                    seq.append(ch)
                q.text = ''.join(seq)
                # Reference behavior: prob < 0.7 -> drop. We keep + flag instead.
                q.flagged = prob < self.min_prob
        return quadrilaterals


class Mit48pxCTCOCR:
    """OCR-CTC fallback recognizer (ocr-ctc.ckpt + alphabet-all-v5.txt).

    Replicates the reference Model48pxCTCOCR recipe: crop height 48px (the
    model's native input, asserted in decode()), batch of 16 sorted by width,
    width padded to (4*(max(widths)+7)//4)+128, greedy CTC top-1 decode.
    Boxes with prob < min_prob are kept + flagged instead of dropped
    (same deviation as the primary OCR, see exp-001).
    """

    name = 'mit48px_ctc'

    def __init__(self, model_dir: str | Path, device: str = 'cuda',
                 min_prob: float = 0.5, padding_pct: float = 0.0):
        self.model_dir = Path(model_dir)
        self.device = device
        self.text_height = 48  # fixed by the model (assert H == 48)
        self.min_prob = min_prob
        self.padding_pct = padding_pct
        self.max_chunk_size = 16

    async def load(self) -> None:
        with open(self.model_dir / 'ocr-ctc' / 'alphabet-all-v5.txt', encoding='utf-8') as fp:
            self.dictionary = [s[:-1] for s in fp.readlines()]
        self.model = CTCOCR(self.dictionary, 768)
        sd = torch.load(self.model_dir / 'ocr-ctc' / 'ocr-ctc.ckpt', map_location='cpu')
        sd = sd['model'] if 'model' in sd else sd
        # reference drops these keys (newer code doesn't use per-layer pe)
        for k in list(sd):
            if k.endswith('.pe.pe'):
                del sd[k]
        self.model.load_state_dict(sd, strict=False)
        self.model.eval()
        if self.device.startswith('cuda'):
            self.model = self.model.cuda()

    async def recognize(self, image: np.ndarray, quadrilaterals: List[Quadrilateral],
                        ) -> List[Quadrilateral]:
        """Fills .text/.prob/.fg_*/.bg_* on each quad (CTC top-1). Returns the list."""
        if not quadrilaterals:
            return quadrilaterals

        region_imgs = [get_region(image, q, q.direction, self.text_height,
                                  self.padding_pct, 1.0, 'none')
                       for q in quadrilaterals]
        perm = sorted(range(len(region_imgs)), key=lambda x: region_imgs[x].shape[1])

        for indices in chunks(perm, self.max_chunk_size):
            N = len(indices)
            widths = [region_imgs[i].shape[1] for i in indices]
            max_width = (4 * (max(widths) + 7) // 4) + 128
            region = np.zeros((N, self.text_height, max_width, 3), dtype=np.uint8)
            for i, idx in enumerate(indices):
                region[i, :, :widths[i], :] = region_imgs[idx]

            images = (torch.from_numpy(region).float() - 127.5) / 127.5
            images = einops.rearrange(images, 'N H W C -> N C H W')
            if self.device.startswith('cuda'):
                images = images.cuda()
            with torch.inference_mode():
                texts = self.model.decode(images, widths, 0)

            for i, single_line in enumerate(texts):
                q = quadrilaterals[indices[i]]
                if not single_line:
                    q.text = ''
                    q.prob = 0.0
                    q.flagged = True
                    continue
                cur_texts = []
                logprobs = []
                frs, fgs, fbs, brs, bgs, bbs = [], [], [], [], [], []
                for (chid, logprob, fr, fg, fb, br, bg, bb) in single_line:
                    ch = self.dictionary[chid]
                    if ch == '<SP>':
                        ch = ' '
                    cur_texts.append(ch)
                    logprobs.append(logprob)
                    if ch != ' ':
                        frs.append(fr); fgs.append(fg); fbs.append(fb)
                        brs.append(br); bgs.append(bg); bbs.append(bb)
                prob = float(np.exp(np.mean(logprobs)))
                q.text = ''.join(cur_texts)
                q.prob = prob
                if frs:
                    q.fg_r = int(np.mean(frs) * 255); q.fg_g = int(np.mean(fgs) * 255)
                    q.fg_b = int(np.mean(fbs) * 255)
                    q.bg_r = int(np.mean(brs) * 255); q.bg_g = int(np.mean(bgs) * 255)
                    q.bg_b = int(np.mean(bbs) * 255)
                q.flagged = prob < self.min_prob
        return quadrilaterals


