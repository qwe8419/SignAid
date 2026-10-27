"""Fingerspelling autocorrect, shared by the Mac app and the standalone Pi.

Edit distance alone picks absurd candidates -- a full system dictionary has
234k entries and will happily turn "helo" into "halo" or "plese" into "plebe".
Ranking by how common a word actually is fixes that, so this ships a compact
frequency table (top ~39k English words, 567KB) rather than depending on the
`wordfreq` package, which does not install cleanly on the Pi.
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
_ALPHA = 'abcdefghijklmnopqrstuvwxyz'

# One edit costs this much zipf frequency. Swept 1.2-3.0 against a
# fingerspelling-error test set: 1.2 scored 11/13 (it let very common two-edit
# words like "that" beat "thank"), everything >= 1.8 scored 12/13.
EDIT_COST = 2.0

# A word is left untouched only if it is at least this common. The corpus
# contains frequent misspellings -- "teh" scores 3.04 -- so a 3.0 cutoff let
# them through uncorrected. Swept below; 3.5 fixes those without touching real
# but uncommon words.
KEEP_MIN_FREQ = 3.5


def _load():
    path = os.path.join(ROOT, 'wordfreq_compact.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


FREQ = _load()
VOCAB = set(FREQ)


def _edits1(word):
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    out = {a + b[1:] for a, b in splits if b}
    out |= {a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1}
    out |= {a + c + b[1:] for a, b in splits if b for c in _ALPHA}
    out |= {a + c + b for a, b in splits for c in _ALPHA}
    return out


def correct_word(word):
    """-> (best, [up to 4 suggestions]). Leaves already-common words alone."""
    w = word.lower()
    if not w or not VOCAB:
        return word, []
    if w in VOCAB and FREQ.get(w, 0) >= KEEP_MIN_FREQ:
        return word, []

    distance = {c: 1 for c in _edits1(w) & VOCAB}
    if len(distance) < 3:
        for e in _edits1(w):
            for c in _edits1(e) & VOCAB:
                distance.setdefault(c, 2)
            if len(distance) > 400:
                break
    if not distance:
        return word, []

    ranked = sorted(distance, key=lambda c: -(FREQ.get(c, 0) - EDIT_COST * distance[c]))
    if w in VOCAB and FREQ.get(w, 0) >= FREQ.get(ranked[0], 0) - 0.5:
        return word, ranked[:4]
    return ranked[0], ranked[:4]


def correct_text(text):
    """Fix a whole buffer. -> (corrected, ["was->now", ...])"""
    out, changes = [], []
    for w in text.split():
        fixed, _ = correct_word(w)
        if fixed.lower() != w.lower():
            changes.append(f"{w}->{fixed}")
        out.append(fixed)
    return ' '.join(out) + (' ' if text.endswith(' ') else ''), changes
