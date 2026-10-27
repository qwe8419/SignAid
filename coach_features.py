"""Hand-shape coach: explain WHAT is wrong with a handshape, in words.

The letter model can say "this is 70% K" but not why, so Practice uses a
separate, interpretable description of the hand: the things a teacher checks.

  curl_thumb .. curl_pinky   total bend of each finger's joints (radians)
  point_down                 which way the INDEX FINGER points: -1 up, 0 sideways,
                             +1 down (the fingers, not the wrist, are what make
                             P and Q point down and G and H point sideways)
  spread_im                  angle between index and middle fingers (radians)
  cross_im                   > 0 normal side by side, < 0 index and middle crossed
  thumb_index                thumb tip to index tip, in hand lengths
  thumb_middle               thumb tip to middle-finger middle joint, hand lengths
  thumb_out                  thumb tip distance from the palm's centre line

For each letter, build_coach.py stores the median and spread of every
measurement over the training data (33k hands, 11 signers). At run time the
measurement furthest from the target letter's typical range becomes the tip.
HandCoach in PracticeView.swift computes exactly these, in the same order;
keep the two in step.

Points are in the model frame (the frame train_robust.py trains in).
"""
import math

import numpy as np

from features import canonicalize, _angle, FINGER_CHAINS, WRIST, MIDDLE_MCP

NAMES = ['curl_thumb', 'curl_index', 'curl_middle', 'curl_ring', 'curl_pinky',
         'point_down', 'spread_im', 'cross_im', 'thumb_index', 'thumb_middle',
         'thumb_out']


def coach_features(raw):
    raw = np.asarray(raw, dtype=np.float64).reshape(21, 3)
    p = canonicalize(raw)
    out = []
    for chain in FINGER_CHAINS:                       # thumb, index, middle, ring, pinky
        bend = sum(math.pi - _angle(p[chain[i]], p[chain[i + 1]], p[chain[i + 2]])
                   for i in range(len(chain) - 2))
        out.append(bend)
    # Direction the index finger points in the (un-rotated) frame, knuckle to
    # tip. Image y grows downward, so +1 is down, -1 up. Mirror-invariant.
    v = raw[8, :2] - raw[5, :2]
    n = np.linalg.norm(v)
    out.append(float(v[1] / n) if n > 1e-9 else 0.0)
    # Index vs middle finger directions, MCP -> tip, in the canonical frame.
    di = p[8, :2] - p[5, :2]
    dm = p[12, :2] - p[9, :2]
    cosang = np.dot(di, dm) / max(np.linalg.norm(di) * np.linalg.norm(dm), 1e-9)
    out.append(float(math.acos(max(-1.0, min(1.0, cosang)))))
    # Crossing: tips in the same left/right order as the knuckles => positive.
    out.append(float(np.sign(p[5, 0] - p[9, 0]) * (p[8, 0] - p[12, 0])))
    out.append(float(np.linalg.norm(p[4] - p[8])))
    out.append(float(np.linalg.norm(p[4] - p[10])))
    # Thumb tip's sideways distance from the wrist->middle-knuckle line.
    out.append(float(abs(p[4, 0])))
    return np.array(out)
