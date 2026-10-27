"""Rebuild the avatar's sign library as REAL 3-D, from the WLASL source videos.

Why this exists
---------------
The old library (avatar_words.json) came from WLASL's published pose keypoints,
which are 2-D. Playback therefore had to invent the third dimension, and no
amount of care makes that convincing: a hand that must touch the chest lands on
roughly the right ray from the shoulder at whatever depth foreshortening
implies, and every fix for one sign breaks another.

MediaPipe's Pose Landmarker returns `pose_world_landmarks` -- metric 3-D, in
metres, origin at the hip centre -- and the Hand Landmarker returns
`hand_world_landmarks`, 3-D in metres about the wrist. Running those over the
original videos recovers the depth the 2-D keypoints threw away. The signer's
hand really did touch their chest, so the avatar's does too, with no body-surface
atlas, no IK, and no hand-authored contact poses.

Disk
----
There is not room to keep 2000 videos, so each one is downloaded, extracted and
deleted before the next starts. Peak usage is one video.

Licensing
---------
WLASL ships under C-UDA-1.0 (tgcn/wlasl-repo/start_kit/C-UDA-1.0.pdf), which
permits computational use of the data. This script's OUTPUT is derived joint
angles, not video: nothing redistributable is copied into the app.

Usage
-----
    python3 authoring/extract_3d.py            # all glosses, resumable
    python3 authoring/extract_3d.py --limit 5  # smoke test
"""
import argparse
import json
import os
import ssl
import sys
import tempfile
import time
import urllib.request

import cv2
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
import mediapipe as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WLASL = os.path.join(ROOT, 'tgcn', 'wlasl-repo', 'start_kit', 'WLASL_v0.3.json')
OUT = os.path.join(ROOT, 'signs_3d.jsonl')
POSE_MODEL = os.path.join(ROOT, 'pose_landmarker.task')
HAND_MODEL = os.path.join(ROOT, 'hand_landmarker.task')

# MediaPipe pose indices for the joints the avatar drives. The rest of the
# 33-point body (legs, feet, face) is not used: signing is upper body, and
# carrying joints nothing reads only makes the file bigger.
POSE_KEEP = {
    'nose': 0, 'l_shoulder': 11, 'r_shoulder': 12, 'l_elbow': 13, 'r_elbow': 14,
    'l_wrist': 15, 'r_wrist': 16, 'l_hip': 23, 'r_hip': 24,
}
POSE_ORDER = list(POSE_KEEP)

# Signing is fast; below this the motion stutters, above it the file bloats.
TARGET_FPS = 20.0
MAX_SECONDS = 4.0
# A clip where the signer is not reliably found is worse than no clip: it plays
# as a limb snapping between detections.
MIN_POSE_RATE = 0.80

_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE
_HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}


def download(url, dest, timeout=25):
    """Fetch one video. Returns True on success."""
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout, context=_ctx) as r:
            if r.status != 200:
                return False
            # Guard against an HTML error page served with a 200.
            ctype = r.headers.get('Content-Type', '')
            if 'video' not in ctype and 'octet-stream' not in ctype:
                return False
            data = r.read(40 * 1024 * 1024)
        if len(data) < 8192:
            return False
        with open(dest, 'wb') as f:
            f.write(data)
        return True
    except Exception:
        return False


class Extractor:
    """Holds the two landmarkers. Creating them is slow, so they are reused."""

    def __init__(self):
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=POSE_MODEL),
                running_mode=vision.RunningMode.VIDEO,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5))
        self.hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=HAND_MODEL),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.25,
                min_tracking_confidence=0.25))
        # MediaPipe's VIDEO mode requires timestamps that never go backwards for
        # the lifetime of the landmarker, and the landmarkers outlive any one
        # clip. A per-video counter restarts at zero and throws.
        self._clock = 0

    def run(self, path, frame_start=1, frame_end=-1):
        """Video -> list of frames, each a dict of metric 3-D landmarks.

        Returns None when the signer is not found reliably enough to be worth
        keeping.
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            cap.release()
            return None

        start = max(int(frame_start) - 1, 0)
        end = total if frame_end in (-1, None) else min(int(frame_end), total)
        if end - start < 4:
            start, end = 0, total

        # Sample at a fixed rate so playback speed matches the real signer
        # rather than being squashed to a fixed frame count, which is what made
        # every sign in the old library take exactly the same time.
        step = max(src_fps / TARGET_FPS, 1.0)
        wanted = []
        i = float(start)
        while i < end and len(wanted) < int(MAX_SECONDS * TARGET_FPS):
            wanted.append(int(i))
            i += step
        if len(wanted) < 4:
            cap.release()
            return None

        frames, found = [], 0
        for n, idx in enumerate(wanted):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            # Upscale small frames: hands are only a few dozen pixels across in
            # a lot of these clips, which is below what the detector will find.
            if max(rgb.shape[:2]) < 640:
                scale = 640.0 / max(rgb.shape[:2])
                rgb = cv2.resize(rgb, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_CUBIC)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self._clock += int(1000.0 / TARGET_FPS)
            ts = self._clock

            body = None
            pr = self.pose.detect_for_video(image, ts)
            if pr.pose_world_landmarks:
                lm = pr.pose_world_landmarks[0]
                # MediaPipe's world frame is y-down; the rig is y-up.
                body = {k: [round(lm[i].x, 4), round(-lm[i].y, 4), round(-lm[i].z, 4)]
                        for k, i in POSE_KEEP.items()}
                found += 1

            left = right = None
            hr = self.hands.detect_for_video(image, ts)
            if hr.hand_world_landmarks:
                for h, handed in zip(hr.hand_world_landmarks, hr.handedness):
                    pts = [[round(p.x, 4), round(-p.y, 4), round(-p.z, 4)] for p in h]
                    # "Left"/"Right" are from the CAMERA's point of view, which
                    # is the signer's mirror. The label already accounts for it.
                    if handed[0].category_name == 'Left':
                        left = pts
                    else:
                        right = pts

            frames.append({'b': body, 'l': left, 'r': right})

        cap.release()
        if not frames or found / len(frames) < MIN_POSE_RATE:
            return None
        return frames


def load_targets():
    with open(WLASL) as f:
        data = json.load(f)
    out = []
    for entry in data:
        # YouTube needs yt-dlp, which is not installed; direct files only.
        urls = [i for i in entry['instances'] if 'youtu' not in i['url']]
        if urls:
            out.append((entry['gloss'], urls))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--tries', type=int, default=4,
                    help='instances to try per gloss before giving up')
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)['gloss'])
                except Exception:
                    pass
    print(f'{len(done)} glosses already extracted', flush=True)

    targets = load_targets()
    if args.limit:
        targets = targets[:args.limit]

    ex = Extractor()
    tmpdir = tempfile.mkdtemp(prefix='wlasl_')
    kept = failed = 0
    t0 = time.time()

    for n, (gloss, instances) in enumerate(targets):
        if gloss in done:
            continue
        frames = None
        for inst in instances[:args.tries]:
            path = os.path.join(tmpdir, 'clip.mp4')
            try:
                if not download(inst['url'], path):
                    continue
                frames = ex.run(path, inst.get('frame_start', 1),
                                inst.get('frame_end', -1))
            finally:
                # Delete before the next attempt: there is no room to keep them.
                if os.path.exists(path):
                    os.remove(path)
            if frames:
                break

        if frames:
            kept += 1
            with open(args.out, 'a') as f:
                f.write(json.dumps({'gloss': gloss, 'fps': TARGET_FPS,
                                    'frames': frames}) + '\n')
        else:
            failed += 1

        if (n + 1) % 10 == 0:
            rate = kept / max(kept + failed, 1)
            el = time.time() - t0
            print(f'[{n+1}/{len(targets)}] kept {kept} failed {failed} '
                  f'({rate:.0%}) {el/60:.1f}min', flush=True)

    print(f'DONE kept={kept} failed={failed}', flush=True)


if __name__ == '__main__':
    main()
