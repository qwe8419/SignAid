"""Pack the extracted 3-D signs into the binary library the app bundles.

signs_3d.jsonl is one line per gloss and is convenient to append to while
extracting, but it is JSON text holding millions of floats. This packs it into
a flat int16 blob plus a small index:

    signs_3d.bin    quantised landmarks, one contiguous run per gloss
    signs_3d.json   {gloss: [offset, frameCount, flags...]}

Quantisation is 1/10000 m over a +/-3.2 m range, which is far finer than the
landmarks are accurate to and about a quarter the size of the text.

Layout, per frame, always the same length so a frame is a fixed stride:
    8 body points   shoulders, elbows, wrists, hips        (l/r each)
   21 left hand points                                     (zeros if absent)
   21 right hand points                                    (zeros if absent)
   = 50 points = 150 int16

A hand that was not detected in a frame is written as zeros and the player holds
the previous shape: handshape changes slowly next to arm motion, so holding is
much less visible than snapping to a default.
"""
import json
import os
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALE = 10000.0
LIMIT = 3.2

BODY_ORDER = ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow',
              'l_wrist', 'r_wrist', 'l_hip', 'r_hip']
POINTS_PER_FRAME = len(BODY_ORDER) + 21 + 21     # 50
VALUES_PER_FRAME = POINTS_PER_FRAME * 3          # 150


def quantise(points):
    a = np.asarray(points, dtype=np.float32)
    a = np.clip(a, -LIMIT, LIMIT)
    return np.rint(a * SCALE).astype(np.int16)


def pack(src, out_bin, out_index):
    blob = []
    index = {}
    offset = 0
    skipped = 0

    with open(src) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            gloss = rec['gloss']
            if gloss in index:
                continue

            rows = []
            last_l = last_r = None
            for fr in rec['frames']:
                body = fr.get('b')
                if not body:
                    # Body is the one thing that cannot be held: it is what
                    # places the arms. Drop the frame instead.
                    continue
                pts = [body[k] for k in BODY_ORDER]
                # Hold the last seen handshape through detection gaps.
                left = fr.get('l') or last_l
                right = fr.get('r') or last_r
                last_l, last_r = left or last_l, right or last_r
                pts += left if left else [[0.0, 0.0, 0.0]] * 21
                pts += right if right else [[0.0, 0.0, 0.0]] * 21
                rows.append(pts)

            if len(rows) < 4:
                skipped += 1
                continue

            arr = quantise(rows).reshape(-1)
            assert arr.size == len(rows) * VALUES_PER_FRAME, gloss
            blob.append(arr)
            index[gloss] = [offset, len(rows)]
            offset += len(rows)

    if not blob:
        print('nothing to pack', file=sys.stderr)
        return

    flat = np.concatenate(blob)
    with open(out_bin, 'wb') as f:
        f.write(flat.tobytes())
    with open(out_index, 'w') as f:
        json.dump({'scale': SCALE, 'stride': VALUES_PER_FRAME,
                   'fps': 20, 'signs': index}, f, separators=(',', ':'))

    mb = flat.nbytes / 1048576
    print(f'{len(index)} signs, {offset} frames, {mb:.1f} MB '
          f'({skipped} skipped for too few usable frames)')


if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'signs_3d.jsonl')
    dest = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'ios', 'SignAid')
    pack(src, os.path.join(dest, 'signs_3d.bin'),
         os.path.join(dest, 'signs_3d.json'))
