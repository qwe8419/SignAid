"""When a recognised letter gets typed: the iPhone app's rules, in Python.

A port of OnDeviceRecognizer.swift (handle(), trackMotion(), isSteady) so the
glasses behave exactly like the phone. The phone counts frames at ~30 fps; the
Pi runs at ~14, so every window here is in SECONDS, set to the phone's frame
counts divided by 30. Same rules, same feel, whatever the frame rate.

  1. Smooth: average the last ~0.17 s of probability vectors (phone: 5 frames).
  2. Vote over the last `window` seconds: the winner must hold >= 60% of the
     frames with mean confidence >= 0.5, over a full window.
  3. Only type while the hand is still (wrist within 0.12 hand-lengths over
     ~0.2 s), except J and Z, which are made of movement.
  4. A held letter types ONCE. The same letter again needs a deliberate act:
     the hand away for 0.5 s, or a slide/bounce of 0.45 hand-lengths that then
     settles (how signers show a double letter). A different letter never needs
     re-arming; it only has to win a full window.
  5. At least 0.35 s between any two letters.
"""
import math
from collections import deque

SPEEDS = {'quick': 8 / 30, 'normal': 12 / 30, 'relaxed': 16 / 30}
SMOOTH_S = 5 / 30
STEADY_S = 6 / 30
MIN_FRAMES = 4            # a "window" of one or two frames is not evidence
ACCEPT_MIN_CONF = 0.50
MIN_AGREEMENT = 0.6
RELEASE_S = 0.5
REPEAT_MOVE = 0.45
STEADY_TOL = 0.12
MIN_GAP_S = 0.35
MOTION_LETTERS = {'J', 'Z'}


class CommitRules:
    def __init__(self, classes, speed='normal'):
        self.classes = list(classes)
        self.window = SPEEDS.get(speed, SPEEDS['normal'])
        self.reset()

    def reset(self):
        self.recent = deque()        # (t, probs)
        self.votes = deque()         # (t, label, conf)
        self.trail = deque()         # (t, x, y)
        self.hand_len = 0.1
        self.last = None
        self.armed = True
        self.away_since = None
        self.anchor = None
        self.moved = False
        self.last_commit_t = -1e9
        self.live, self.live_conf, self.progress = None, 0.0, 0.0

    # -------------------------------------------------------------- motion --
    def _steady(self):
        if len(self.trail) < 3:
            return False
        mx = sum(p[1] for p in self.trail) / len(self.trail)
        my = sum(p[2] for p in self.trail) / len(self.trail)
        spread = max(math.hypot(p[1] - mx, p[2] - my) for p in self.trail)
        return spread / self.hand_len <= STEADY_TOL

    def _track(self, pts, now):
        wx, wy = pts[0][0], pts[0][1]
        length = math.hypot(pts[9][0] - wx, pts[9][1] - wy)
        if length > 1e-3:
            self.hand_len = length
        self.trail.append((now, wx, wy))
        # Keep at least 3 points: at the Pi's ~14 fps, 0.2 s is only 2-3 frames.
        while len(self.trail) > 3 and now - self.trail[0][0] > STEADY_S:
            self.trail.popleft()
        if not self.armed and self.anchor is not None:
            if math.hypot(wx - self.anchor[0], wy - self.anchor[1]) / self.hand_len >= REPEAT_MOVE:
                self.moved = True
            if self.moved and self._steady():
                self.armed = True
                self.votes.clear()        # the repeat must earn a full window

    # ---------------------------------------------------------------- step --
    def step(self, probs, pts, now):
        """One frame. probs: class probabilities or None (no hand); pts: the
        21 (x, y, z) landmarks in frame coordinates. -> letter to type, or None."""
        if probs is None:
            self.live, self.live_conf, self.progress = None, 0.0, 0.0
            self.recent.clear()
            if self.away_since is None:
                self.away_since = now
            if now - self.away_since >= RELEASE_S:
                self.armed = True
                self.trail.clear()
            return None
        self.away_since = None
        self._track(pts, now)

        self.recent.append((now, probs))
        while self.recent and now - self.recent[0][0] > SMOOTH_S:
            self.recent.popleft()
        n = len(self.recent)
        avg = [sum(p[i] for _, p in self.recent) / n for i in range(len(self.classes))]
        best = max(range(len(avg)), key=avg.__getitem__)
        label, conf = self.classes[best], avg[best]
        self.live, self.live_conf = label, conf

        self.votes.append((now, label, conf))
        while self.votes and now - self.votes[0][0] > self.window:
            self.votes.popleft()
        weight, count = {}, {}
        for _, l, c in self.votes:
            weight[l] = weight.get(l, 0.0) + c
            count[l] = count.get(l, 0) + 1
        winner = max(weight, key=weight.get)
        mean_conf = weight[winner] / count[winner]
        agreement = count[winner] / len(self.votes)
        span = now - self.votes[0][0]
        full = len(self.votes) >= MIN_FRAMES and span >= self.window * 0.85

        is_repeat = winner == self.last
        self.progress = 0.0 if (is_repeat and not self.armed) else \
            min(span / self.window, 1.0) * agreement
        qualifies = full and mean_conf >= ACCEPT_MIN_CONF and agreement >= MIN_AGREEMENT
        steady = self._steady() or winner in MOTION_LETTERS
        if not (qualifies and steady and (not is_repeat or self.armed)
                and now - self.last_commit_t >= MIN_GAP_S):
            return None

        self.last, self.armed = winner, False
        self.last_commit_t = now
        self.anchor = (self.trail[-1][1], self.trail[-1][2]) if self.trail else None
        self.moved = False
        self.votes.clear()
        self.progress = 0.0
        return winner

    def typed_manually(self, letter, now):
        """Mirror of the phone's append(): a tapped letter counts as typed."""
        self.last, self.armed, self.moved = letter, False, False
        self.last_commit_t = now
        self.anchor = (self.trail[-1][1], self.trail[-1][2]) if self.trail else None
