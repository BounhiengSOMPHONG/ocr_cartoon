#!/usr/bin/env python3
"""Run the bundled page OCR on one image.

    python example.py page.jpg                      # print boxes + text
    python example.py page.jpg --out page.jsonl     # also write the JSONL
    python example.py page.jpg --device cpu --rtl   # CPU / right-to-left page
"""
import argparse
import sys
from pathlib import Path

from manga_page_ocr import PageOCR, load_image_rgb, save_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('image', help='page image (jpg/png/webp)')
    ap.add_argument('--out', help='write regions as JSONL to this path')
    ap.add_argument('--device', default=None, help="'cuda' (default) or 'cpu'")
    ap.add_argument('--rtl', action='store_true', help='right-to-left reading order')
    ap.add_argument('--model-dir', default=None,
                    help='model directory (default: models/ next to the package)')
    args = ap.parse_args()

    ocr = PageOCR(model_dir=args.model_dir, device=args.device,
                  right_to_left=args.rtl)
    regions = ocr.run_sync(load_image_rgb(args.image), Path(args.image).name)

    for r in regions:
        flag = f'  <- {r.flag}' if r.flag else ''
        print(f'[{r.id:02d}] {r.confidence:.3f} {r.direction:>10}  {r.text}{flag}')

    accepted = [r for r in regions if not r.flag]
    review = [r for r in regions if r.flag == 'needs_review']
    low = [r for r in regions if r.flag == 'low_confidence']
    print(f'\n{len(regions)} boxes: {len(accepted)} accepted, '
          f'{len(low)} low-confidence, {len(review)} needs_review', file=sys.stderr)

    if args.out:
        save_jsonl(regions, args.out)
        print(f'wrote {len(regions)} regions -> {args.out}', file=sys.stderr)


if __name__ == '__main__':
    main()
