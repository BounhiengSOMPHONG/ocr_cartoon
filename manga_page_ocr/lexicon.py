"""Word-frequency evidence for choosing between two recognizer outputs.

Every Latin box is read twice — once by the beam model (MIT48px) and once by
the CTC fallback — and on a handwritten font the two often disagree in a way
confidence alone cannot settle: one of them reads English words and the other
reads a blur.

    'SOMEONEELSE' 0.947  vs  CTC 'SOMEONE ELSE'  1.000
    'BECOUSE'     0.975  vs  CTC 'Because'        0.887
    'WOLDNTCAL'   1.000  vs  CTC "WOULDN'T CALL"  0.994

The exp-105 rule took the higher confidence, which picks the CTC in exactly one
of those three (and it compares letters only, so the first pair — same letters,
different spacing — was invisible to it). This module answers the question the
confidence was standing in for: which of these two readings is made of English
words? (exp-117)

The table is a word -> zipf frequency map derived from wordfreq (provenance and
licence in data/ATTRIBUTION.md). A token counts only at MIN_TOKEN_LEN letters
or more: 'I', 'OF', 'TO' read the same either way and carry no evidence.

Deliberately *not* used to respell words. A distance-1 spelling corrector over
this same table was measured against the project's ground truth and mangles
sound effects and names — 'GLUB GLUB GURGLE GLUB' -> 'CLUB CLUB CLUB',
'BRANNER' -> 'BANNER', 'AIKA' -> 'AKA' — so the lexicon only ever chooses
between texts a recognizer actually produced (exp-117).
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

WORD_FREQ_PATH = (Path(__file__).resolve().parent / 'data'
                  / 'en-word-frequency.tsv')

MIN_TOKEN_LEN = 4
# zipf: 7.7 = 'the', 5.0 = 'morning', 3.0 = 'ringtone', 2.4 = 'sekai' (romaji),
# 2.1 = 'becouse' (a misspelling the corpus knows), 0 = not a word at all.
# Below ~2.5 a token is rare enough that a *reading* built from it is worth
# second-guessing; a correct reading ('ringtone' 2.99) stays above it.
SUSPICIOUS_BELOW = 2.5

_TOKEN = re.compile(r'[A-Za-z]+')


@lru_cache(maxsize=1)
def _frequency() -> dict:
    table: dict = {}
    with open(WORD_FREQ_PATH, encoding='utf-8') as fp:
        for line in fp:
            word, _, zipf = line.partition('\t')
            if zipf:
                table[word] = float(zipf)
    return table


def _token_scores(text: str) -> list:
    tokens = [t.lower() for t in _TOKEN.findall(text or '')]
    tokens = [t for t in tokens if len(t) >= MIN_TOKEN_LEN]
    freq = _frequency()
    return [freq.get(t, 0.0) for t in tokens]


def word_score(text: str) -> float | None:
    """Mean zipf frequency of the text's content words — how English it reads.

    None when the text has no token long enough to judge (an empty string, a
    lone punctuation mark, '<UNK>'): then there is nothing to prefer.
    """
    scores = _token_scores(text)
    return sum(scores) / len(scores) if scores else None


def weakest_token(text: str) -> float | None:
    """Lowest zipf among the text's content words — its weakest link.

    This, not the mean, is what opens the guard below. A mean hides one bad
    word behind a good one: 'IF IT REALLY BECOMESF' — the beam model's reading
    of a line the CTC reads correctly — averages 2.99 and would have been
    defended, while its weakest token (0.0) says plainly that 'BECOMESF' is
    not a word. One unknown token is the whole signal.
    """
    scores = _token_scores(text)
    return min(scores) if scores else None


def reads_more_like_english(primary: str, alternative: str) -> bool:
    """True when `alternative` is the reading made of (real) English words.

    Two steps, because either alone is wrong (exp-117):

    - the guard: the primary has to contain a word the language disowns. Only
      then is its place up for debate; a correct rare word keeps its place
      even when a common one is on offer ('BLONDIE' 3.03 vs a misread
      'BLONDE' 4.4), which is the failure mode that made plain
      confidence-swapping wrong.
    - the choice: between two readings that both contain an unknown token, the
      one that reads more like English on average wins.
    """
    if not alternative or alternative == primary:
        return False
    weakest = weakest_token(primary)
    if weakest is None or weakest >= SUSPICIOUS_BELOW:
        return False
    p, a = word_score(primary), word_score(alternative)
    return a is not None and p is not None and a > p
