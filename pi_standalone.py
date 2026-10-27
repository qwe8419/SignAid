"""SignAid running entirely ON THE GLASSES. No Mac.

Pipeline: webcam -> MediaPipe hand landmarks -> 298 geometric features -> a
small MLP -> letter. Serves the same HTTP API as the Mac app, so the iPhone app
points at the Pi instead and nothing else changes.

Design choices that make this fit on a Pi 4:

* No PyTorch. The net is 298->128->64->26 with the BatchNorms folded into the
  preceding Linear layers, so inference is three numpy matmuls. Verified
  identical to the torch model (max abs diff 1.7e-06, same argmax on every
  sample). Saves a ~200MB dependency.
* No YOLO. The crop classifier costs ~100-200ms on a Pi and needs the image.
  Landmarks alone score 0.949 in EVERY lighting condition (the crop model swings
  0.097-0.981), so dropping it costs ~3 points in good light and buys both speed
  and lighting independence.
* No word model. The TGCN needs 50 pose frames; on a Pi that is ~30s per sign.
  Words stay on the Mac build.

Run:
    ~/signaid/venv/bin/python3 pi_standalone.py
"""
import argparse
import json
import os
import queue
import subprocess
import threading
import time
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, request
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from autocorrect import correct_text, correct_word
from commit_rules import CommitRules
from features import features_from_landmarks

ROOT = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

ACCEPT_MIN_CONF = float(os.environ.get('ACCEPT_MIN_CONF', 0.50))
MIN_AGREEMENT = float(os.environ.get('MIN_AGREEMENT', 0.5))
AUTO_STABLE_FRAMES = int(os.environ.get('AUTO_STABLE_FRAMES', 6))
AUTO_RELEASE_FRAMES = int(os.environ.get('AUTO_RELEASE_FRAMES', 4))
# MediaPipe releases the GIL, so worker THREADS give real parallelism without
# the cost of shipping 900KB frames between processes. Measured on the Pi 4:
# 1 thread 8.1 fps, 2 -> 12.8, 3 -> 16.3, 4 -> 17.1. Three is the sweet spot;
# the fourth barely helps and starves the camera and Flask of a core.
WORKERS = int(os.environ.get('WORKERS', 3))
# Tracking mode (default): ONE worker running MediaPipe in VIDEO mode, which
# finds the palm once and then only tracks it, instead of re-detecting it in
# every frame on three threads. Measured on this Pi 4: the 3-worker IMAGE mode
# held all four cores, hit 82 C, throttled, and fell to ~8 fps with 200 ms
# gaps -- visible stutter in the app. TRACKING=0 restores the old mode.
TRACKING = os.environ.get('TRACKING', '1') == '1'
# Camera-only mode (default). The iPhone runs MediaPipe and the letter model on
# the /camera stream itself: a Pi 4 needs ~120 ms per frame for hand tracking
# even in VIDEO mode, ran at 82 C and throttled, and stuttered; the phone does
# the same work at 30 fps. PI_INFERENCE=1 brings back on-Pi recognition (for
# the old /detect, /state and /landmarks routes).
PI_INFERENCE = os.environ.get('PI_INFERENCE', '0') == '1'
camera_fps = {'fps': 0.0}
if TRACKING:
    WORKERS = 1

# ---------------------------------------------------------------- model ----

class LetterNet:
    """A 3-seed ensemble, each member three matmuls with BatchNorm folded in.

    Averaging three seeds costs ~0.15 ms total and buys about a point of
    robustness under stress perturbation (0.927 vs 0.918 for one net), which is
    worth having now that this is the only letter model on the glasses.
    """

    def __init__(self, path):
        z = np.load(path, allow_pickle=True)
        self.classes = [str(c) for c in z['classes']]
        # float32 throughout: mean/std come out of numpy as float64, which
        # would promote every matmul to float64 -- measured 206 ms/frame on the
        # Pi versus 120 ms in float32, because it loses SIMD.
        f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)
        self.mean, self.std = f32(z['mean']), f32(z['std'])
        self.n = int(z['n_models']) if 'n_models' in z else 1
        self.layers = [tuple(f32(z[f'{name}_{k}']) for name in
                             ('w0', 'b0', 'w1', 'b1', 'w2', 'b2'))
                       for k in range(self.n)]

    def probs(self, feats):
        x = ((np.asarray(feats, dtype=np.float32) - self.mean) / self.std)
        total = None
        for w0, b0, w1, b1, w2, b2 in self.layers:
            h = np.maximum(x @ w0.T + b0, 0)
            h = np.maximum(h @ w1.T + b1, 0)
            z = h @ w2.T + b2
            e = np.exp(z - z.max())
            p = e / e.sum()
            total = p if total is None else total + p
        return total / self.n


# The robust model (train_robust.py: 11 signers, mirror- and camera-invariant)
# when present, else the original webcam-only one. Same tensor layout, same
# 300 features, so nothing else changes. LETTER_MODEL=... overrides.
_model = os.environ.get('LETTER_MODEL') or next(
    p for p in (os.path.join(ROOT, 'letter_net_weights_robust.npz'),
                os.path.join(ROOT, 'letter_net_weights.npz')) if os.path.exists(p))
net = LetterNet(_model)
print(f"Loaded letter net: {len(net.classes)} classes, "
      f"{net.n}-model ensemble")

def make_detector():
    """One per worker thread -- a HandLandmarker instance is not thread-safe."""
    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=os.path.join(ROOT, 'hand_landmarker.task')),
            running_mode=(vision.RunningMode.VIDEO if TRACKING
                          else vision.RunningMode.IMAGE),
            num_hands=1))

# The 2000-sign motion file is ~40MB. Loading it at startup would delay the
# camera by seconds on a Pi, so it is read on first use instead.
_avatar_lock = threading.Lock()
avatar_library = None
avatar_words = None


def load_avatar():
    global avatar_library, avatar_words
    with _avatar_lock:
        if avatar_library is None:
            try:
                with open(os.path.join(ROOT, 'avatar_library.json')) as f:
                    avatar_library = json.load(f)
                with open(os.path.join(ROOT, 'avatar_words.json')) as f:
                    avatar_words = json.load(f)
                print(f"Avatar ready: {len(avatar_words)} signs")
            except FileNotFoundError:
                # Do NOT cache the failure: the files may be copied over while
                # the server is already running, and caching {} here would make
                # it 404 forever until restarted.
                print("WARNING: avatar files missing on the Pi")
                return {}, {}
    return avatar_library, avatar_words


# ---------------------------------------------------------------- state ----

current_text = ''
latest_frame = None
frame_seq = 0            # bumped per captured frame so the preview can skip re-encoding
latest_annotated = None
lock = threading.Lock()
auto_enabled = False
auto_state = {'candidate': None, 'count': 0, 'idle': 0, 'armed': True,
              'last': None, 'live': None, 'live_conf': 0.0}
perf = {'fps': 0.0, 'infer_ms': 0.0}
# Latest hand points for /landmarks. The iPhone app runs the letter model and
# typing rules itself, on these, so the glasses and the phone share one code
# path. A Condition so each stream wakes the moment a new frame lands.
landmark_cv = threading.Condition()
landmark_latest = {'seq': 0, 'pts': None}
# The most recent recognition result, kept apart from the frame it was computed
# on. The preview draws it over the CURRENT frame instead of showing the older
# frame it belonged to -- otherwise the video lags inference by 200-400 ms and
# signing feels like moving underwater.
overlay = {'box': None, 'label': None, 'conf': 0.0, 'at': 0.0}
# Rolling history of recent predictions, filled by the collector. /detect votes
# over these instead of running its own inference: with 3 workers already
# saturating the cores, spinning up a 4th MediaPipe instance and doing 9 more
# detections starved itself and made the request hang for 30s+.
history = deque(maxlen=12)
# Small queues on purpose: depth 2 keeps latency near one frame time.
work_q = queue.Queue(maxsize=WORKERS + 1)
result_q = queue.Queue(maxsize=WORKERS + 1)


def classify(frame, detector):
    """-> (label, confidence) or None when no hand is visible."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    if TRACKING:
        # VIDEO mode needs strictly increasing timestamps; one worker, in order.
        classify.ts = max(int(time.monotonic() * 1000), getattr(classify, 'ts', 0) + 1)
        result = detector.detect_for_video(image, classify.ts)
    else:
        result = detector.detect(image)
    if not result.hand_landmarks:
        return None, None
    lms = result.hand_landmarks[0]
    p = net.probs(features_from_landmarks(lms))
    i = int(p.argmax())
    return (net.classes[i], float(p[i]), p), lms


def capture_loop(device, width, height):
    """Only grabs frames. Inference happens on the worker threads."""
    global latest_frame
    # Wait for the camera rather than giving up. At boot the service starts
    # before a USB webcam has enumerated, and raising here killed only this
    # thread -- the server stayed up reporting camera_connected=false forever.
    while True:
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        # MJPG from the webcam: 30 fps at 640x480 over USB, where raw YUYV
        # can drop to 10-15 at the same size.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if cap.isOpened() and cap.read()[0]:
            break
        cap.release()
        print(f"camera {device} not ready, retrying in 2s", flush=True)
        time.sleep(2)
    # Hold 30 fps in dim rooms. Logitech webcams otherwise halve the frame rate
    # for longer exposures (measured: 15 fps indoors), and signing smears.
    subprocess.run(['v4l2-ctl', '-d', f'/dev/video{device}',
                    '-c', 'exposure_dynamic_framerate=0'], capture_output=True)
    seq = 0
    n, t0 = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02)
            continue
        n += 1
        if time.time() - t0 >= 2.0:
            camera_fps['fps'] = n / (time.time() - t0)
            n, t0 = 0, time.time()
        frame = cv2.flip(frame, 1)
        global frame_seq
        with lock:
            latest_frame = frame
            frame_seq += 1
        seq += 1
        try:
            work_q.put_nowait((seq, frame))
        except queue.Full:
            # Drop rather than queue: on glasses a stale frame is worse than a
            # missing one, and a backed-up queue only adds latency.
            pass


def worker_loop():
    detector = make_detector()
    while True:
        seq, frame = work_q.get()
        t = time.time()
        pred, lms = classify(frame, detector)
        ms = (time.time() - t) * 1000
        try:
            result_q.put_nowait((seq, frame, pred, lms, ms))
        except queue.Full:
            pass


def collector_loop():
    """Single consumer, so the auto-detect state machine stays sequential.

    Workers finish out of order, so anything older than what we have already
    consumed is discarded instead of rewinding the letter buffer."""
    last_seq = 0
    frames = 0
    t0 = time.time()
    while True:
        seq, frame, pred, lms, ms = result_q.get()
        if seq <= last_seq:
            continue                      # stale: a newer frame already landed
        last_seq = seq

        if lms is not None:
            h, w = frame.shape[:2]
            xs = [p.x * w for p in lms]
            ys = [p.y * h for p in lms]
            box = (max(0, int(min(xs)) - 40), max(0, int(min(ys)) - 40),
                   min(w, int(max(xs)) + 40), min(h, int(max(ys)) + 40))
        else:
            box = None
        now = time.time()
        overlay.update(box=box, label=pred[0] if pred else None,
                       conf=pred[1] if pred else 0.0, at=now)
        if pred:
            history.append((pred[0], pred[1], now))
        with landmark_cv:
            landmark_latest['seq'] += 1
            landmark_latest['pts'] = ([[round(p.x, 5), round(p.y, 5), round(p.z, 5)]
                                       for p in lms] if lms is not None else None)
            landmark_cv.notify_all()

        if auto_enabled:
            step_auto(pred, lms, now)
        else:
            auto_state.update(live=None, live_conf=0.0, candidate=None, count=0)
            rules.reset()

        frames += 1
        if time.time() - t0 >= 2.0:
            perf['fps'] = frames / (time.time() - t0)
            perf['infer_ms'] = ms
            frames = 0
            t0 = time.time()


# The iPhone's typing rules (commit_rules.py), so a letter behaves the same on
# the glasses as on the phone: typed once while held, repeated only by sliding
# or lowering the hand, never while the hand is moving. LETTER_SPEED matches
# the app's Settings ▸ Letter speed: quick, normal or relaxed.
rules = CommitRules(net.classes, os.environ.get('LETTER_SPEED', 'normal'))


def step_auto(pred, lms, now):
    global current_text
    pts = [(p.x, p.y, p.z) for p in lms] if lms is not None else None
    letter = rules.step(pred[2] if pred else None, pts, now)
    auto_state.update(live=rules.live, live_conf=rules.live_conf,
                      progress=rules.progress)
    if letter:
        current_text += letter


def vote(num_frames=9, delay=0.05, max_age=1.5):
    """Vote over the predictions the collector has already produced.

    Deliberately runs no inference of its own. The previous version created a
    fourth HandLandmarker and ran 9 detections while three worker threads were
    already using every core, which made /detect hang for 30 seconds.
    """
    deadline = time.time() + 2.0
    while len(history) < 3 and time.time() < deadline:
        time.sleep(0.05)          # give the workers a moment on a cold start

    now = time.time()
    recent = [(l, c) for (l, c, t) in list(history)[-num_frames:] if now - t <= max_age]
    if not recent:
        return None, [], 0.0

    weights, confs = Counter(), {}
    for label, conf in recent:
        weights[label] += conf
        confs.setdefault(label, []).append(conf)

    winner, _ = weights.most_common(1)[0]
    mean_conf = sum(confs[winner]) / len(confs[winner])
    agreement = len(confs[winner]) / len(recent)
    breakdown = [(l, len(confs[l]), sum(confs[l]) / len(confs[l]))
                 for l, _ in weights.most_common()]
    if mean_conf < ACCEPT_MIN_CONF or agreement < MIN_AGREEMENT:
        return None, breakdown, mean_conf
    return winner, breakdown, mean_conf


# ---------------------------------------------------------------- routes ---

@app.route('/camera')
def camera():
    """Every new frame, full 640x480, no overlay: the input the iPhone runs
    hand tracking and recognition on. Mirrored (selfie view), which the letter
    model does not care about -- it is trained on both hands."""
    def gen():
        last_sent = -1
        while True:
            with lock:
                f = latest_frame if frame_seq != last_sent else None
                seq = frame_seq
            if f is None:
                time.sleep(0.004)
                continue
            last_sent = seq
            ok, buf = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + buf.tobytes() + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/landmarks')
def landmarks():
    """One JSON line per processed frame: {"pts": [[x, y, z] x21] or null}.
    Coordinates are normalised to the mirrored 640x480 frame -- the frame the
    letter model was trained in, so the app applies no rotation."""
    def gen():
        seen = 0
        while True:
            with landmark_cv:
                landmark_cv.wait_for(lambda: landmark_latest['seq'] != seen, timeout=2.0)
                seen = landmark_latest['seq']
                pts = landmark_latest['pts']
            yield json.dumps({'pts': pts}) + '\n'
    return Response(gen(), mimetype='application/x-ndjson',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/health')
def health():
    return jsonify({'app': 'SignAid (standalone Pi)', 'source': 'pi-camera',
                    'camera_connected': latest_frame is not None,
                    'frame_age': 0.0, 'letters': True, 'words': False,
                    'avatar_vocabulary': len(avatar_words) if avatar_words else 2000,
                    'auto': auto_enabled,
                    'fps': round(perf['fps'] if PI_INFERENCE else camera_fps['fps'], 1),
                    'infer_ms': round(perf['infer_ms'], 1),
                    'workers': WORKERS})


@app.route('/video')
def video():
    def gen():
        last_sent = -1
        last_time = 0.0
        # The preview is for a human, not the recogniser: ~12 fps at 480px looks
        # fine on a phone and leaves the cores for inference. Encoding every
        # captured frame at full size cost 6 fps of recognition when measured.
        min_interval = 1.0 / 12
        while True:
            if time.time() - last_time < min_interval:
                time.sleep(0.005)
                continue
            with lock:
                if latest_frame is None or frame_seq == last_sent:
                    f = None
                else:
                    f = latest_frame.copy()
                    last_sent = frame_seq
            if f is None:
                # No new frame yet. Re-encoding the same one just burns CPU that
                # the inference workers need -- that cost 6 fps when measured.
                time.sleep(0.01)
                continue
            # Overlay the latest result onto the CURRENT frame. It is drawn only
            # while fresh, so a stale box does not linger over a moved hand.
            o = dict(overlay)
            if o['box'] and time.time() - o['at'] < 0.4:
                x1, y1, x2, y2 = o['box']
                cv2.rectangle(f, (x1, y1), (x2, y2), (0, 255, 136), 2)
                if o['label']:
                    cv2.putText(f, f"{o['label']} {o['conf']:.0%}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 136), 2)
            if f.shape[1] > 480:
                f = cv2.resize(f, (480, int(480 * f.shape[0] / f.shape[1])),
                               interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if ok:
                last_time = time.time()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + buf.tobytes() + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detect', methods=['POST'])
def detect():
    global current_text
    winner, breakdown, conf = vote()
    votes = ' | '.join(f"{n}:{c}({p:.2f})" for n, c, p in breakdown)
    if winner:
        current_text += winner
        return jsonify({'detected': winner, 'candidates': f"Votes: {votes}",
                        'word': current_text,
                        'message': f"Added: {winner} ({conf:.0%})"})
    if breakdown:
        g = breakdown[0]
        return jsonify({'detected': '?', 'candidates': f"Votes: {votes}",
                        'word': current_text,
                        'message': f"Unsure — best guess {g[0]} at {g[2]:.0%}"})
    return jsonify({'detected': None, 'candidates': '', 'word': current_text,
                    'message': 'No hand detected'})


@app.route('/detect_word', methods=['POST'])
def detect_word():
    return jsonify({'detected': None, 'candidates': '', 'word': current_text,
                    'message': 'Word signs need the Mac build (TGCN is too slow here)'})


@app.route('/auto', methods=['POST'])
def set_auto():
    global auto_enabled
    auto_enabled = bool((request.get_json(silent=True) or {}).get('enabled'))
    auto_state.update(candidate=None, count=0, armed=True, last=None)
    rules.reset()
    return jsonify({'enabled': auto_enabled})


@app.route('/state')
def state():
    words = current_text.split()
    suggestions = correct_word(words[-1])[1] if words else []
    return jsonify({'word': current_text, 'auto': auto_enabled,
                    'live': auto_state['live'],
                    'live_conf': round(auto_state['live_conf'], 2),
                    'progress': auto_state.get('progress', 0.0),
                    'suggestions': suggestions})


@app.route('/space', methods=['POST'])
def space():
    global current_text
    if current_text and not current_text.endswith(' '):
        current_text += ' '
    return jsonify({'word': current_text})


@app.route('/delete', methods=['POST'])
def delete():
    global current_text
    current_text = current_text[:-1]
    return jsonify({'word': current_text})


@app.route('/clear', methods=['POST'])
def clear():
    global current_text
    current_text = ''
    return jsonify({'word': current_text})


@app.route('/autocorrect', methods=['POST'])
def autocorrect_route():
    global current_text
    current_text, changes = correct_text(current_text)
    return jsonify({'word': current_text,
                    'message': 'Corrected: ' + ', '.join(changes) if changes
                               else 'Nothing to correct'})


@app.route('/apply_suggestion', methods=['POST'])
def apply_suggestion():
    global current_text
    choice = (request.get_json(silent=True) or {}).get('word', '')
    words = current_text.split()
    if words and choice:
        words[-1] = choice
        current_text = ' '.join(words) + ' '
    return jsonify({'word': current_text})


@app.route('/avatar/library')
def avatar_lib():
    lib, _ = load_avatar()
    return (jsonify(lib), 200) if lib else (jsonify({'error': 'not built'}), 404)


@app.route('/avatar/word/<gloss>')
def avatar_word(gloss):
    _, words = load_avatar()
    if not words or gloss not in words:
        return jsonify({'error': 'unknown gloss'}), 404
    return jsonify({'gloss': gloss, 'frames': words[gloss]})


@app.route('/translate', methods=['POST'])
def translate():
    """English -> avatar segments. Known glosses play a real recorded sign;
    everything else is fingerspelled, which is what a signer does for names."""
    lib, words = load_avatar()
    text = ((request.get_json(silent=True) or {}).get('text') or '').strip()
    if not text or not words:
        return jsonify({'segments': [], 'unknown': [],
                        'message': 'Type something first' if words
                                   else 'Avatar files missing'})
    letters = lib.get('letters', {})
    segments, unknown = [], []
    for raw in text.split():
        token = ''.join(ch for ch in raw if ch.isalnum()).lower()
        if not token:
            continue
        if token in words:
            segments.append({'kind': 'word', 'label': token})
            continue
        spelled = [c.upper() for c in token if c.upper() in letters]
        if spelled:
            segments.append({'kind': 'fingerspell', 'label': raw, 'letters': spelled})
        else:
            unknown.append(raw)
    n_signed = sum(1 for s in segments if s['kind'] == 'word')
    msg = f"{n_signed} signed, {len(segments) - n_signed} fingerspelled"
    if unknown:
        msg += f" — no sign for: {', '.join(unknown[:5])}"
    return jsonify({'segments': segments, 'unknown': unknown, 'message': msg})


@app.route('/speak', methods=['POST'])
def speak():
    global current_text
    spoken = current_text
    if spoken:
        # espeak if present; silently skip if the glasses have no audio
        subprocess.Popen(['espeak', spoken],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    current_text = ''
    return jsonify({'spoken': spoken})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--port', type=int, default=8080)
    args = ap.parse_args()

    threading.Thread(target=capture_loop,
                     args=(args.device, args.width, args.height),
                     daemon=True).start()
    if PI_INFERENCE:
        for _ in range(WORKERS):
            threading.Thread(target=worker_loop, daemon=True).start()
        threading.Thread(target=collector_loop, daemon=True).start()
    time.sleep(3)
    print(f"SignAid glasses on port {args.port}: "
          + (f"{WORKERS} recognition workers" if PI_INFERENCE
             else "camera only, recognition runs on the iPhone"))
    app.run(host='0.0.0.0', port=args.port, threaded=True)


if __name__ == '__main__':
    main()
