"""Build the hand-shape coach's per-letter checks, and test them.

    python3 build_coach.py            # report + write ios/SignAid/letter_coach.json

Each letter has a short list of the measurements that DEFINE it (what a
teacher would check: "index and middle straight, ring and pinky folded, fingers
apart, pointing up" for V). The acceptable range of each comes from real
examples of the letter: the 5th to 95th percentile over 33k training hands from
11 signers. When a check fails, the coach reports the one missed by the most,
measured in that check's own spread, as a plain-language tip.

The tests: on confusions learners actually make (signing K when asked for P, U
for V, ...), does the tip name the real difference? And on correct examples of
the target, how often does the coach stay quiet?
"""
import json
import os
from collections import Counter

import numpy as np

from coach_features import NAMES, coach_features
from train_robust import load_all

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'ios', 'SignAid', 'letter_coach.json')
LO, HI = 3, 97
# A check must be missed by this many spreads before it becomes a tip, so a
# hand at the edge of normal is not "corrected".
MARGIN = 0.5
# Checks that only mean something in one direction. Crossing is a fault for U
# only when the fingers ARE crossed; spread fingertips also move the value, and
# that is spread_im's job. For R the fault is NOT crossing.
ONE_SIDED = {('R', 'cross_im'): 'high', ('U', 'cross_im'): 'low'}

FINGERS = ['curl_index', 'curl_middle', 'curl_ring', 'curl_pinky']
CHECKS = {
    'A': FINGERS + ['thumb_out'],
    'B': FINGERS + ['spread_im', 'curl_thumb'],
    'C': FINGERS + ['thumb_index'],
    'D': FINGERS,
    'E': FINGERS + ['curl_thumb'],
    'F': ['thumb_index', 'curl_middle', 'curl_ring', 'curl_pinky'],
    'G': FINGERS + ['point_down'],
    'H': FINGERS + ['point_down', 'spread_im'],
    'I': ['curl_pinky', 'curl_index', 'curl_middle', 'curl_ring'],
    'K': FINGERS + ['point_down', 'spread_im', 'thumb_middle'],
    'L': FINGERS + ['thumb_out'],
    'M': FINGERS + ['thumb_out'],
    'N': FINGERS + ['thumb_out'],
    'O': ['thumb_index', 'curl_index', 'curl_middle', 'curl_ring', 'curl_pinky'],
    'P': ['point_down', 'curl_index', 'curl_ring', 'curl_pinky', 'thumb_middle'],
    'Q': ['point_down', 'curl_middle', 'curl_ring', 'curl_pinky'],
    'R': FINGERS + ['cross_im'],
    'S': FINGERS + ['thumb_out'],
    'T': FINGERS + ['thumb_out'],
    'U': FINGERS + ['spread_im', 'cross_im', 'point_down'],
    'V': FINGERS + ['spread_im', 'point_down'],
    'W': FINGERS,
    'X': FINGERS,
    'Y': ['curl_pinky', 'curl_index', 'curl_middle', 'curl_ring', 'thumb_out'],
}
# Minimum spread, so a letter always signed one exact way does not turn a
# small wobble into a correction.
FLOOR = {'curl': 0.3, 'point_down': 0.25, 'spread_im': 0.08, 'cross_im': 0.06,
         'thumb': 0.1}


def floor(name):
    return next(v for k, v in FLOOR.items() if name.startswith(k))


def build(F, Y):
    table = {}
    for L, names in CHECKS.items():
        X = F[Y == L]
        checks = []
        for n in names:
            v = X[:, NAMES.index(n)]
            lo, hi = np.percentile(v, LO), np.percentile(v, HI)
            scale = max((np.percentile(v, 75) - np.percentile(v, 25)) / 1.35, floor(n))
            only = ONE_SIDED.get((L, n), 'both')
            if only == 'high':
                lo = -1e9                     # only flag values above the range
            if only == 'low':
                hi = 1e9
            checks.append({'feature': n, 'lo': round(float(lo), 4),
                           'hi': round(float(hi), 4), 'scale': round(float(scale), 4),
                           'median': round(float(np.median(v)), 4)})
        table[L] = checks
    return table


def advise(f, checks):
    """-> (feature, +1 too high / -1 too low) for the worst miss, or None."""
    worst, best = None, 0.0
    for c in checks:
        v = f[NAMES.index(c['feature'])]
        miss = (v - c['hi']) / c['scale'] if v > c['hi'] else \
               (c['lo'] - v) / c['scale'] if v < c['lo'] else 0.0
        if miss > MARGIN and miss > best:
            best, worst = miss, (c['feature'], 1 if v > c['hi'] else -1)
    return worst


def main():
    data = load_all()
    P = np.concatenate([d[0] for d in data.values()])
    Y = np.concatenate([d[1] for d in data.values()])
    F = np.vstack([coach_features(p) for p in P])
    table = build(F, Y)

    quiet = [np.mean([advise(f, table[L]) is None for f in F[Y == L]]) for L in table]
    print(f'correct signing left alone: {np.mean(quiet):.0%} of frames '
          f'(worst letter {min(quiet):.0%})')

    # (target, what was signed, acceptable tips: any feature starting with one)
    expect = [
        ('P', 'K', ['point_down']), ('Q', 'G', ['point_down']),
        ('H', 'U', ['point_down']), ('G', 'L', ['thumb_out', 'point_down']),
        ('V', 'U', ['spread_im']), ('U', 'V', ['spread_im']),
        ('R', 'U', ['cross_im']), ('U', 'R', ['cross_im']),
        ('B', 'A', ['curl_']), ('D', 'A', ['curl_index']), ('L', 'A', ['curl_index']),
        ('Y', 'A', ['curl_pinky', 'thumb_out']), ('I', 'A', ['curl_pinky']),
        ('W', 'V', ['curl_ring']), ('F', 'B', ['thumb_index', 'curl_index']),
        ('O', 'C', ['thumb_index']), ('V', 'W', ['curl_ring']), ('A', 'B', ['curl_']),
    ]
    print(f'\n{"target":>6} {"signed":>6} {"right tip":>10}   most common tip')
    scores = []
    for target, signed, ok in expect:
        tips = [advise(f, table[target]) for f in F[Y == signed]]
        named = [t[0] for t in tips if t]
        good = np.mean([any(n.startswith(o) for o in ok) for n in named]) if named else 0.0
        common = Counter(named).most_common(1)[0][0] if named else '-'
        print(f'{target:>6} {signed:>6} {good:>10.0%}   {common}')
        scores.append(good)
    print(f'\naverage: {np.mean(scores):.0%} of tips name the right fix')

    with open(OUT, 'w') as f:
        json.dump({'features': NAMES, 'letters': table}, f)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
