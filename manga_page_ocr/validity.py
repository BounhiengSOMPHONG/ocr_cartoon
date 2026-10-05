"""Text validity checks for the Phase-3 quality gate (plan Phase 3 checklist):

  - empty text
  - abnormal repeated characters
  - symbol ratio too high
  - text implausibly long for the bubble size
  - detector found a region but OCR returned implausible text
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

_SYMBOLS = set('!?.,:;"\'()-…~#*$%&@+=<>[]{}|/\\^_')
_CJK = re.compile(r'[　-鿿぀-ヿ＀-￯]')
# Control tokens of the recognizer's alphabet — '<UNK>' is the model saying it
# cannot read the crop (its alphabet is English, so a Korean sound effect ends
# up here). They are not text: drawn on the page they put the literal string
# over the artwork instead of the letters the user was trying to read.
_CONTROL = re.compile(r'<(?:UNK|SEP|LF|PAD|UNUSED\d*)>', re.IGNORECASE)


def clean_control_tokens(text: str) -> Tuple[str, bool]:
    """Drop alphabet control tokens. Returns (text, whether any were found)."""
    if not text or '<' not in text:
        return text or '', False
    cleaned = _CONTROL.sub(' ', text)
    if cleaned == text:
        return text, False
    return re.sub(r'\s+', ' ', cleaned).strip(), True


def check_text(text: str, language: str = 'en',
               max_repeat: int = 4, max_symbol_ratio: float = 0.5,
               ) -> Tuple[bool, Optional[str]]:
    """Returns (valid, reason)."""
    if not text or not text.strip():
        return False, 'empty'
    if re.search(r'(.)\1{%d,}' % max_repeat, text):
        return False, 'repeated_chars'
    # leading/trailing punctuation is normal in manga ("...SO", "HUH?!") —
    # measure the symbol ratio on the inner text only
    core = text
    while core and core[0] in _SYMBOLS:
        core = core[1:]
    while core and core[-1] in _SYMBOLS:
        core = core[:-1]
    if not core:
        return False, 'empty'
    n_sym = sum(1 for c in core if c in _SYMBOLS or c.isspace())
    if n_sym / len(core) > max_symbol_ratio:
        return False, 'symbol_ratio'
    if language == 'en' and _CJK.search(text):
        # neighbor CJK leaking into the crop (seen with ctd boxes in exp-101)
        return False, 'cjk_in_en'
    return True, None


def looks_latin(text: str, min_letters: int = 4, ratio: float = 0.7) -> bool:
    """True when the letters of `text` are mostly not CJK.

    Used the other way round from `cjk_in_en`: a box the script gate sent to
    the *Japanese* backend that comes back as Latin letters was almost
    certainly misrouted — manga-ocr reads a Latin crop as (fullwidth) Latin
    rather than inventing kana, so its own output is the evidence. Other
    non-CJK scripts count here too; they are equally wrong for that backend.
    """
    letters = [c for c in text if c.isalpha()]
    if len(letters) < min_letters:
        return False
    non_cjk = sum(1 for c in letters if not _CJK.match(c))
    return non_cjk / len(letters) >= ratio


def check_length(text: str, box_w: int, box_h: int, language: str = 'en',
                 ) -> Tuple[bool, Optional[str]]:
    """Rough check: can this much text fit in the box?

    Assumes average glyph width ~0.55 * text height (original image px).
    More than ~1.5x the fitted capacity is flagged.
    """
    if box_h <= 0:
        return True, None
    glyph_w = 0.55 * box_h
    capacity = max(1, int(box_w / glyph_w))
    if len(text) > 1.5 * capacity:
        return False, 'too_long_for_box'
    return True, None
