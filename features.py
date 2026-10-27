"""Shared landmark feature extraction for the confusable-letter classifier.

Both the collectors, the trainer and app.py must use `extract_features` so the
model always sees the same feature layout. `FEATURE_VERSION` is stored in the
pickle and checked at load time.
"""
import numpy as np

FEATURE_VERSION = 3

# MediaPipe hand landmark indices
WRIST = 0
MIDDLE_MCP = 9
FINGER_CHAINS = [
    [0, 1, 2, 3, 4],       # thumb
    [0, 5, 6, 7, 8],       # index
    [0, 9, 10, 11, 12],    # middle
    [0, 13, 14, 15, 16],   # ring
    [0, 17, 18, 19, 20],   # pinky
]
TIPS = [4, 8, 12, 16, 20]


def landmarks_to_array(landmarks):
    """MediaPipe landmark list -> (21, 3) float array."""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)


def canonicalize(pts):
    """Translate to wrist, scale by hand size, rotate so wrist->middle-MCP
    points 'up'. Removes position, scale and in-plane rotation."""
    pts = np.asarray(pts, dtype=np.float64).reshape(21, 3)
    pts = pts - pts[WRIST]

    scale = np.linalg.norm(pts[MIDDLE_MCP, :2])
    if scale < 1e-6:
        scale = max(np.linalg.norm(pts[:, :2], axis=1).max(), 1e-6)
    pts = pts / scale

    v = pts[MIDDLE_MCP, :2]
    n = np.linalg.norm(v)
    if n > 1e-6:
        # rotate v onto (0, -1) i.e. up in image coords
        cos_a, sin_a = -v[1] / n, -v[0] / n
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        pts[:, :2] = pts[:, :2] @ rot.T
    return pts


def _angle(a, b, c):
    """Angle at b in the triangle a-b-c, in radians."""
    v1, v2 = a - b, c - b
    d = np.linalg.norm(v1) * np.linalg.norm(v2)
    if d < 1e-9:
        return 0.0
    return float(np.arccos(np.clip(np.dot(v1, v2) / d, -1.0, 1.0)))


def extract_features(pts):
    """(21,3) landmarks (raw or already wrist-normalized) -> 1-D feature vector.

    Ends with two ORIENTATION features (sin/cos of the raw hand angle before
    canonicalisation). Canonicalising rotation is what makes the other 298
    features robust to a tilted camera, but it also discards the only thing
    separating letters that share a handshape and differ in direction:
    H vs U are the same shape 49 degrees apart, and G vs Q likewise. sin/cos
    is used rather than the angle itself to avoid the wrap-around at +/-180.
    """
    raw = np.asarray(pts, dtype=np.float64).reshape(21, 3)
    v = raw[MIDDLE_MCP, :2] - raw[WRIST, :2]
    hand_angle = np.arctan2(v[1], v[0])
    p = canonicalize(pts)
    feats = [p.ravel()]

    # all pairwise 3-D distances (210) — pose-invariant shape descriptor
    diff = p[:, None, :] - p[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(21, k=1)
    feats.append(dist[iu])

    # joint angles along each finger (3 per finger = 15)
    angles = []
    for chain in FINGER_CHAINS:
        for i in range(len(chain) - 2):
            angles.append(_angle(p[chain[i]], p[chain[i + 1]], p[chain[i + 2]]))
    feats.append(np.array(angles))

    # fingertip spread: angle of each tip about the wrist (5) + tip heights (5)
    tips = p[TIPS]
    feats.append(np.arctan2(tips[:, 1], tips[:, 0]))
    feats.append(tips[:, 1])
    feats.append(np.array([np.sin(hand_angle), np.cos(hand_angle)]))

    return np.concatenate(feats)


def features_from_landmarks(landmarks):
    """Convenience wrapper for MediaPipe output."""
    return extract_features(landmarks_to_array(landmarks))


def augment(pts, rng, rot_deg=18.0, scale_jit=0.12, noise=0.015, z_noise=0.03):
    """Random rigid + noise perturbation of a (21,3) landmark set."""
    pts = np.asarray(pts, dtype=np.float64).reshape(21, 3).copy()

    a = np.deg2rad(rng.uniform(-rot_deg, rot_deg))
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    c = pts[WRIST, :2].copy()
    pts[:, :2] = (pts[:, :2] - c) @ rot.T + c

    pts *= rng.uniform(1 - scale_jit, 1 + scale_jit)
    pts[:, :2] += rng.normal(0, noise, size=(21, 2))
    pts[:, 2] += rng.normal(0, z_noise, size=21)
    return pts


FEATURE_DIM = len(extract_features(np.random.RandomState(0).rand(21, 3)))
