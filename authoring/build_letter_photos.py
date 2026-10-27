"""Pick one real photo per letter for fingerspelling.

The avatar's rigged hand was inferring finger curl from landmark spacing, which
is an approximation of a shape that is the entire content of the letter. These
are photographs of the hand actually making it.

Selection is the MEDOID, not an average: the sample whose landmarks sit closest
to every other sample of that letter. Averaging landmarks produces a pose that
never occurred — fingers half-bent between two real positions — whereas the
medoid is always a hand someone actually made. Straight-on captures are
preferred, since a letter photographed at an angle reads as a different letter.

Source is letter_dataset/, captured locally, so unlike the sign videos there is
no licensing question here at all.
"""
import json
import os
import shutil

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from PIL import Image, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'letter_dataset')
OUT = os.path.join(ROOT, 'letter_photos')
SIZE = 320          # square
# Fraction of the hand's own size left as margin around it. The stored crops
# were cut at whatever the detector returned, so they vary in zoom; letters
# flashing past at different magnifications reads as the image jumping.
#
# Kept tight on purpose. A wider box gives a nicer composition and pulls the
# signer's FACE into most frames, which is a picture of a real person shipped
# inside an app. The hand is the whole content of a fingerspelled letter.
MARGIN = 0.20


def square_crop(frame_path, detector):
    """Crop the full frame to a consistent square around the hand.

    Returns None if the hand is not found, in which case the caller falls back
    to the stored crop.
    """
    bgr = cv2.imread(frame_path)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.hand_landmarks:
        return None
    h, w = rgb.shape[:2]
    xs = [p.x * w for p in res.hand_landmarks[0]]
    ys = [p.y * h for p in res.hand_landmarks[0]]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half = max(max(xs) - min(xs), max(ys) - min(ys)) * (0.5 + MARGIN)
    x0, y0 = int(cx - half), int(cy - half)
    x1, y1 = int(cx + half), int(cy + half)
    im = Image.fromarray(rgb)
    # Crop can run past the frame edge; pad rather than shrink the box, so the
    # hand stays the same size in every letter.
    return im.crop((x0, y0, x1, y1))


def canonical(landmarks):
    """Wrist-relative, scale-normalised, so distance compares SHAPE only."""
    p = np.asarray(landmarks, dtype=np.float64).reshape(21, 3)
    p = p - p[0]
    scale = np.linalg.norm(p[9, :2])
    if scale < 1e-6:
        scale = max(np.linalg.norm(p[:, :2], axis=1).max(), 1e-6)
    return (p / scale).ravel()


def main():
    df = pd.read_csv(os.path.join(DATA, 'index.csv'))
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(OUT):
        for f in os.listdir(OUT):
            os.remove(os.path.join(OUT, f))

    detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=os.path.join(ROOT, 'hand_landmarker.task')),
            running_mode=vision.RunningMode.IMAGE, num_hands=1,
            min_hand_detection_confidence=0.3))

    chosen = {}
    for letter, group in df.groupby('letter'):
        # Prefer head-on captures; fall back to everything if there are none.
        straight = group[group['variation'].astype(str).str.contains(
            'straight', case=False, na=False)]
        pool = straight if len(straight) >= 5 else group

        feats, rows = [], []
        for _, row in pool.iterrows():
            try:
                feats.append(canonical(json.loads(row['landmarks_json'])))
                rows.append(row)
            except Exception:
                continue
        if not feats:
            continue

        X = np.vstack(feats)
        # Medoid: smallest total distance to all other samples.
        d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
        best = rows[int(d.sum(axis=1).argmin())]

        im = square_crop(os.path.join(DATA, best['frame_path']), detector)
        if im is None:
            src = os.path.join(DATA, best['crop_path'])
            if not os.path.exists(src):
                continue
            im = Image.open(src).convert('RGB')
        im = ImageOps.pad(im.convert('RGB'), (SIZE, SIZE), color=(16, 17, 22),
                          method=Image.LANCZOS)
        im.save(os.path.join(OUT, f'{letter.upper()}.jpg'), quality=88)
        chosen[letter.upper()] = {'n': len(rows), 'src': best['crop_path']}

    total = sum(os.path.getsize(os.path.join(OUT, f))
                for f in os.listdir(OUT))
    print(f'{len(chosen)} letters -> {OUT} ({total/1024:.0f} KB)')
    missing = sorted(set('ABCDEFGHIJKLMNOPQRSTUVWXYZ') - set(chosen))
    if missing:
        print('no photo for:', ' '.join(missing))


if __name__ == '__main__':
    main()
