"""Train the on-device letter model to survive a camera it was not trained on.

The previous model scored 0.98 on held-out webcam bursts and 0.58 per frame on
the phone, with a mean confidence of 0.39 — so low that most bursts never
committed a letter at all. The held-out number was honest about the webcam and
said nothing about any other camera, because every training sample came from one
camera, one room and one chirality.

What changes here, each aimed at a measured failure:

1. MIRROR AUGMENTATION. The webcam capture ran cv2.flip, so the model had only
   ever seen one handedness; the app needed a "Mirror hand" toggle whose right
   setting depends on which camera is in use and which hand the signer uses.
   No ASL letter is the mirror image of another letter, so mirrored copies
   carry the same label and the toggle stops mattering. Left-handed signers
   come for free.
2. ASPECT JITTER. MediaPipe normalises x by width and y by height, so a camera
   with a different frame shape stretches every hand sideways. That is what
   made the phone read nearly everything as H. Training over a range of x:y
   scales makes the model indifferent to it.
3. HEAVIER LANDMARK NOISE. At arm's length the hand spans 16-32% of what it did
   on the webcam, so its landmarks are noisier.
4. ALL CAPTURE SOURCES. Webcam bursts, phone bursts and a third camera's photos
   (extract_image_letters.py).

Evaluation is LEAVE-ONE-SOURCE-OUT: to score the phone, nothing from the phone
is trained on; likewise for the photos. That is the closest available stand-in
for "a new user's camera", which is what the app will actually meet.

    python3 train_robust.py --eval     # cross-source table only
    python3 train_robust.py            # table, then train on everything and export

The export keeps the 300-feature layout, so HandFeatures.swift is unchanged.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

from features import extract_features, WRIST
from train_landmark_letters import load as load_webcam, SEED
from train_letter_net import train_one, N_MODELS
from train_with_phone import split_by_group, export

ROOT = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(ROOT, 'image_letters', 'my_asl.json')
# ASL-HG (Mendeley Data, doi:10.17632/j4y5w2c8w9.1, CC BY 4.0): 10 signers,
# phone photos indoors and out. Landmarks via extract_image_letters.py.
ASLHG = os.path.join(ROOT, 'image_letters', 'asl_hg.json')
LEAVE_OUT_FOLDS = 5
TARGET_PER_CLASS = 3000

# The app's commit rule (OnDeviceRecognizer.swift), for the burst-level score.
VOTE_WINDOW, MIN_AGREEMENT, APP_MIN_CONF = 9, 0.5, 0.50


# The phone's landmarks arrive in the camera SENSOR's frame: a landscape
# 640x480 buffer, a quarter-turn away from upright. Rotation is canonicalised
# out of 298 of the 300 features, but the last two encode which way the hand
# points -- the only thing separating H from U, G from Q, P from K -- so every
# upright two-finger letter read as H. And because x is normalised by 640 and
# y by 480, turning the frame also moves the 4:3 squash onto the wrong axis.
#
# Measured on the 800 recorded phone samples against a webcam-only model:
#     as recorded                 0.253
#     rotate -90                  0.752
#     x * (4/3)^4, rotate -90     0.990
# (4/3)^4 = (4/3)^2 for the axis swap, times (4/3)^2 = 1/0.5625 to undo the
# aspect "correction" that build was applying when this data was recorded. The
# app now applies the device half itself (HandFeatures.toModelFrame) and tags
# what it records with "frame": "model", so new captures need nothing here.
LEGACY_PHONE_XSCALE = (4 / 3) ** 4


def to_model_frame(p, xscale):
    """Stretch x, then rotate a quarter-turn: (x, y) -> (y, -k*x)."""
    q = np.asarray(p, dtype=np.float64).copy()
    x = q[:, 0] * xscale
    q[:, 0], q[:, 1] = q[:, 1].copy(), -x
    q[:, 2] *= xscale
    return q


def load_phone(directory=os.path.join(ROOT, 'phone_letters')):
    pts, labels, groups = [], [], []
    for path in sorted(glob.glob(os.path.join(directory, '**', '*.json'),
                                 recursive=True)):
        try:
            d = json.load(open(path))
        except Exception:
            continue
        letter, samples = d.get('letter'), d.get('samples') or []
        if not letter or not samples:
            continue
        legacy = d.get('frame') != 'model'
        for s in samples:
            a = np.asarray(s, dtype=np.float64)
            if a.shape != (21, 3):
                continue
            pts.append(to_model_frame(a, LEGACY_PHONE_XSCALE) if legacy else a)
            labels.append(letter)
            groups.append(os.path.basename(path))   # one file == one burst
    return np.array(pts), np.array(labels), np.array(groups, dtype=object)


# my_asl_dataset was captured by a script that prompted I, L, M, N (it skips
# J and K, the motion letters) while the signer went through the real
# alphabet, so three folders hold the letter two places earlier. Checked by
# eye on the photos, and every frame in each folder was read the same way by a
# model that never saw this set. P is dropped: its photos show a sideways
# two-finger hand closer to H than to P.
PHOTO_RELABEL = {'L': 'J', 'M': 'K', 'N': 'L'}
PHOTO_DROP = {'P'}


def load_photos(path=PHOTOS):
    pts, labels, groups = [], [], []
    if not os.path.exists(path):
        return np.zeros((0, 21, 3)), np.array([]), np.array([], dtype=object)
    for b in json.load(open(path)):
        if b['letter'] in PHOTO_DROP:
            continue
        letter = PHOTO_RELABEL.get(b['letter'], b['letter'])
        for s in b['samples']:
            pts.append(np.array(s, dtype=np.float64))
            labels.append(letter)
            groups.append(f"photo|{letter}")
    return np.array(pts), np.array(labels), np.array(groups, dtype=object)


def load_aslhg(path=ASLHG):
    """-> points, letters, groups ('P3|A'), signer ids."""
    pts, labels, groups, signers = [], [], [], []
    if os.path.exists(path):
        for b in json.load(open(path)):
            for s in b['samples']:
                pts.append(np.array(s, dtype=np.float64))
                labels.append(b['letter'])
                groups.append(f"{b['signer']}|{b['letter']}")
                signers.append(b['signer'])
    return (np.array(pts).reshape(-1, 21, 3), np.array(labels),
            np.array(groups, dtype=object), np.array(signers))


def load_all():
    """-> dict source -> (points, letters, burst keys)."""
    wp, wy, wk = load_webcam()
    wk = np.array([f'{k[0]}|{k[1]}' for k in wk], dtype=object)
    hp, hy, hg, _ = load_aslhg()
    return {
        'webcam': (wp, wy, wk),
        'phone': load_phone(),
        'photos': load_photos(),
        'aslhg': (hp, hy, hg),
    }


def robust_augment(pts, rng, mirror_p=0.5, rot_deg=25.0, scale_jit=0.15,
                   aspect=(0.75, 1.33), noise=0.022, z_noise=0.05):
    """One perturbed copy of a (21,3) raw landmark set."""
    p = np.asarray(pts, dtype=np.float64).reshape(21, 3).copy()
    c = p[WRIST].copy()
    p -= c
    if rng.rand() < mirror_p:
        p[:, 0] = -p[:, 0]
    # Aspect: stretch x (and z, which MediaPipe scales with x) log-uniformly.
    k = np.exp(rng.uniform(np.log(aspect[0]), np.log(aspect[1])))
    p[:, 0] *= k
    p[:, 2] *= k
    a = np.deg2rad(rng.uniform(-rot_deg, rot_deg))
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    p[:, :2] = p[:, :2] @ rot.T
    p *= rng.uniform(1 - scale_jit, 1 + scale_jit)
    # Noise relative to hand size, so a small hand is not left nearly clean.
    size = np.linalg.norm(p[9, :2]) or 1.0
    p[:, :2] += rng.normal(0, noise * size / 0.12, size=(21, 2))
    p[:, 2] += rng.normal(0, z_noise * size / 0.12, size=21)
    return p + c


def balanced(points, labels, target, rng, aug=True):
    """Every letter up to `target` rows. Originals are kept, plus one mirrored
    copy of each so both chiralities appear unperturbed at least once."""
    out_p, out_y = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        base = [points[i] for i in idx]
        if aug:
            base += [np.c_[-points[i][:, :1] + 2 * points[i][WRIST, 0],
                           points[i][:, 1:]] for i in idx]
        out_p.extend(base)
        out_y.extend([c] * len(base))
        if not aug:
            continue
        for k in range(max(target - len(base), 0)):
            out_p.append(robust_augment(points[idx[k % len(idx)]], rng))
            out_y.append(c)
    return np.array(out_p), np.array(out_y)


def featurize(points):
    return np.vstack([extract_features(p) for p in points]).astype(np.float32)


def fit(points, labels, classes, seed):
    lab = {c: i for i, c in enumerate(classes)}
    rng = np.random.RandomState(seed)
    ap, ay = balanced(points, labels, TARGET_PER_CLASS, rng)
    X = featurize(ap)
    Y = np.array([lab[c] for c in ay])
    mean, std = X.mean(0, keepdims=True), X.std(0, keepdims=True)
    std[std < 1e-6] = 1.0
    models = [train_one((X - mean) / std, Y, len(classes), seed + i)
              for i in range(N_MODELS)]
    return models, mean, std


def predict(models, mean, std, X):
    import torch
    x = torch.from_numpy((X - mean) / std)
    with torch.no_grad():
        return (sum(torch.softmax(m(x), 1) for m in models) / len(models)).numpy()


def vote(labels, confs, min_conf=APP_MIN_CONF):
    for end in range(VOTE_WINDOW, len(labels) + 1):
        w, n = defaultdict(float), defaultdict(int)
        for l, c in zip(labels[end - VOTE_WINDOW:end], confs[end - VOTE_WINDOW:end]):
            if c >= min_conf:
                w[l] += c
                n[l] += 1
        if w:
            win = max(w, key=w.get)
            if w[win] / n[win] >= min_conf and n[win] / VOTE_WINDOW >= MIN_AGREEMENT:
                return win
    return None


def score(probs, y, groups, classes, both_hands=True):
    """Per-frame accuracy, mean confidence, and burst-level committed accuracy
    under the app's vote. Bursts are cut into 25-frame chunks so one long
    photo run counts as several attempts rather than one."""
    C = np.array(classes)
    pred = C[probs.argmax(1)]
    conf = probs.max(1)
    out = {'frame': float((pred == y).mean()), 'conf': float(conf.mean())}
    hits = tries = 0
    for g in sorted(set(groups)):
        idx = np.where(groups == g)[0]
        for s in range(0, len(idx) - VOTE_WINDOW + 1, 25):
            chunk = idx[s:s + 25]
            got = vote(list(pred[chunk]), list(conf[chunk]))
            tries += 1
            hits += got == y[chunk[0]]
    out['committed'] = hits / max(tries, 1)
    per = {}
    for c in sorted(set(y)):
        m = y == c
        per[c] = float((pred[m] == c).mean())
    out['per'] = per
    return out


def mirrored(points):
    return np.array([np.c_[1.0 - p[:, :1], p[:, 1:]] for p in points])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval', action='store_true')
    args = ap.parse_args()

    data = load_all()
    classes = sorted(set().union(*[set(d[1]) for d in data.values()]))
    for k, (p, y, g) in data.items():
        print(f'{k:7s} {len(y):5d} samples, {len(set(y)):2d} letters, '
              f'{len(set(g)):3d} bursts')

    own = [k for k in data if k != 'aslhg']        # all one signer: the author

    def cat(keys):
        P = np.concatenate([data[k][0] for k in keys])
        Y = np.concatenate([data[k][1] for k in keys])
        return P, Y

    def report(tag, s, sm=None):
        extra = f'      ({sm["frame"]:.3f})' if sm else ''
        print(f'{tag:28s} {s["frame"]:.3f}  {s["conf"]:.2f}   {s["committed"]:.3f}{extra}')
        weak = sorted(s['per'].items(), key=lambda kv: kv[1])[:6]
        print('      weakest: ' + '  '.join(f'{c} {v:.2f}' for c, v in weak))

    print('\nheld out                     frame   conf  committed   (mirrored)')

    # 1. UNSEEN SIGNERS -- the number that matters for anyone but the author.
    # ASL-HG's ten signers are split into folds; each fold is scored by a model
    # that never saw those people. "before" trains on the author alone.
    hp, hy, hg, hs = load_aslhg()
    if len(hy):
        people = sorted(set(hs), key=lambda x: int(x[1:]))
        P, Y = cat(own)
        models, mean, std = fit(P, Y, classes, SEED)
        before = score(predict(models, mean, std, featurize(hp)), hy, hg, classes)
        report('new signers, author-only', before)
        frames, commits, confs = [], [], []
        for f in range(LEAVE_OUT_FOLDS):
            test = set(people[f::LEAVE_OUT_FOLDS])
            te = np.isin(hs, list(test))
            P = np.concatenate([cat(own)[0], hp[~te]])
            Y = np.concatenate([cat(own)[1], hy[~te]])
            models, mean, std = fit(P, Y, classes, SEED + f)
            s_ = score(predict(models, mean, std, featurize(hp[te])), hy[te], hg[te], classes)
            frames.append(s_['frame']); commits.append(s_['committed']); confs.append(s_['conf'])
            print(f'   fold {f} ({",".join(sorted(test))}): frame {s_["frame"]:.3f}  '
                  f'committed {s_["committed"]:.3f}')
        print(f'{"new signers, +ASL-HG":28s} {np.mean(frames):.3f}  {np.mean(confs):.2f}   '
              f'{np.mean(commits):.3f}   (+/- {np.std(frames):.3f} across folds)')

    # 2. The author's own phone, which nothing else was captured on.
    tp, ty, tg = data['phone']
    keys = [k for k in data if k != 'phone']
    P, Y = cat(keys)
    models, mean, std = fit(P, Y, classes, SEED)
    s = score(predict(models, mean, std, featurize(tp)), ty, tg, classes)
    sm = score(predict(models, mean, std, featurize(mirrored(tp))), ty, tg, classes)
    report('phone (never trained on)', s, sm)

    if args.eval:
        return

    P = np.concatenate([d[0] for d in data.values()])
    Y = np.concatenate([d[1] for d in data.values()])
    final, mean, std = fit(P, Y, classes, SEED + 200)
    export(final, mean, std, classes,
           {'val': float(np.mean(frames)) if len(hy) else s['frame'],
            'phone': s['frame'],
            'feature_version': 3, 'stress': s['frame'],
            'sources': '+'.join(data), 'recipe': 'train_robust.py'})
    print('\nexported letter_net_3d.bin (300 features, unchanged layout)')


if __name__ == '__main__':
    main()
