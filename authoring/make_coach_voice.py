"""Record the Practice audio coach's voice, once, on this Mac.

    python3 authoring/make_coach_voice.py --model <dir with kokoro-v1.0.onnx
                                           and voices-v1.0.bin> [--voice af_heart]

The coach says a fixed set of phrases, so they are synthesised ahead of time
and bundled (coach_voice/, ~3 MB) instead of calling a cloud voice at run time.
That keeps Practice offline and private, the same promise as the rest of the
app, with no API key in the binary and no per-use cost.

Voice: Kokoro-82M (hexgrad, Apache-2.0), run through kokoro-onnx. Apache-2.0
permits bundling and commercial use. Model files:
https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0

Phrases that name a letter are generated whole ("Next one, K.") rather than
stitched from a "Next one" clip and a "K" clip, so the intonation is natural.
Several wordings of each kind are recorded and the app picks at random, so the
coach does not repeat itself every few seconds.
"""
import argparse
import json
import os
import subprocess
import tempfile

import soundfile as sf
from kokoro_onnx import Kokoro

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'coach_voice')
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
PRACTICE = 'ABCDEFGHIKLMNOPQRSTUVWXY'      # J and Z need motion; not practised

# kind -> list of wordings. {L} is the letter.
PHRASES = {
    'welcome': ["Welcome to practice! I'll say a letter, and you sign it.",
                "Hi there! Let's practice some fingerspelling together."],
    'audio_on': ["Audio coach on."],
    'first': ["Let's start with {L}.", "First up, {L}."],
    'next': ["Next one, {L}.", "Now try {L}.", "How about {L}?", "Okay, {L}."],
    'praise': ["Great!", "Nice job!", "Perfect!", "You got it!", "Awesome!",
               "That's it!", "Beautiful!", "Well done!", "Excellent!", "Yes, nailed it!"],
    'encourage': ["Keep going, you've got this.", "Almost there!",
                  "You're close. Keep trying.", "Take your time.",
                  "Hold it nice and steady."],
    'looks_like': ["That looks like {L}."],
    'try': ["Try {L}.", "We're going for {L}."],
    'no_hand': ["I can't see your hand. Bring it up to the camera.",
                "Whenever you're ready, raise your hand."],
    'skip': ["No problem, let's skip that one.", "Let's try a different one."],
    'streak_5': ["Five in a row!"],
    'streak_10': ["Ten in a row! You're on fire!"],
    'streak_20': ["Twenty in a row! That's amazing!"],
}
# Hand-shape coach tips (build_coach.py / HandCoach in PracticeView.swift).
# Keys must match HandCoach.tipText.
TIPS = {
    'tip_curl_thumb_straighten': "Straighten your thumb out.",
    'tip_curl_thumb_fold': "Fold your thumb across your palm.",
    'tip_curl_index_straighten': "Straighten your index finger.",
    'tip_curl_index_fold': "Fold your index finger down.",
    'tip_curl_middle_straighten': "Straighten your middle finger.",
    'tip_curl_middle_fold': "Fold your middle finger down.",
    'tip_curl_ring_straighten': "Straighten your ring finger.",
    'tip_curl_ring_fold': "Fold your ring finger down.",
    'tip_curl_pinky_straighten': "Straighten your pinky.",
    'tip_curl_pinky_fold': "Fold your pinky down.",
    'tip_point_up': "Point your fingers straight up.",
    'tip_point_down': "Point your hand down.",
    'tip_point_side': "Turn your hand sideways, with your fingers pointing across.",
    'tip_point_forward': "Point your index finger forward, not straight down.",
    'tip_spread_together': "Press your index and middle fingers together.",
    'tip_spread_apart': "Spread your index and middle fingers apart.",
    'tip_cross_do': "Cross your middle finger over your index finger.",
    'tip_cross_undo': "Uncross your fingers.",
    'tip_thumbindex_touch': "Touch your thumb to your index finger.",
    'tip_thumbindex_away': "Move your thumb away from your index finger.",
    'tip_thumbmiddle_touch': "Rest your thumb against your middle finger.",
    'tip_thumbmiddle_away': "Move your thumb off your middle finger.",
    'tip_thumbout_in': "Bring your thumb in against your hand.",
    'tip_thumbout_out': "Stick your thumb out to the side.",
}
# How to make each letter, read when you ask for help. Keys match
# HandCoach.howTo in the app.
HOW = {
    'A': "Make a fist, with your thumb resting against the side of your index finger.",
    'B': "Hold your four fingers straight up and together, with your thumb folded across your palm.",
    'C': "Curve your fingers and thumb into a C, like you're holding a cup.",
    'D': "Point your index finger up, and curve your other fingers to touch your thumb.",
    'E': "Bend all four fingertips down to rest on your thumb, tucked across your palm.",
    'F': "Touch your index finger to your thumb in a circle, and hold your other three fingers up.",
    'G': "Point your index finger sideways, with your thumb just above it. Fold the other fingers.",
    'H': "Point your index and middle fingers sideways, side by side.",
    'I': "Make a fist and raise your pinky straight up.",
    'K': "Raise your index and middle fingers in a V, with your thumb resting between them.",
    'L': "Point your index finger up and your thumb out to the side, making an L.",
    'M': "Make a fist with your thumb tucked under your first three fingers.",
    'N': "Make a fist with your thumb tucked under your first two fingers.",
    'O': "Curve all your fingertips to touch your thumb, making an O.",
    'P': "Make a K, then turn it so your hand points down.",
    'Q': "Make a G, then turn it so your finger and thumb point down.",
    'R': "Cross your middle finger over your index finger, both pointing up.",
    'S': "Make a fist with your thumb across the front of your fingers.",
    'T': "Make a fist with your thumb tucked between your index and middle fingers.",
    'U': "Hold your index and middle fingers straight up, pressed together.",
    'V': "Hold your index and middle fingers up and apart, in a V.",
    'W': "Hold your index, middle and ring fingers up and spread, with your thumb holding your pinky.",
    'X': "Raise your index finger and bend it into a hook. Fold the other fingers.",
    'Y': "Stick out your thumb and your pinky, and fold the other three fingers.",
}
for key, text in TIPS.items():
    PHRASES[key] = [text]
for L, text in HOW.items():
    PHRASES[f'how_{L}'] = [f"Here's how to sign {L}. " + text]
PHRASES['help_intro'] = ["Let me help.", "Here's a tip."]

PER_LETTER = {'first': PRACTICE, 'next': PRACTICE, 'try': PRACTICE, 'looks_like': LETTERS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--voice', default='af_heart')
    ap.add_argument('--speed', type=float, default=1.0)
    args = ap.parse_args()

    k = Kokoro(os.path.join(args.model, 'kokoro-v1.0.onnx'),
               os.path.join(args.model, 'voices-v1.0.bin'))
    os.makedirs(OUT, exist_ok=True)
    manifest = {'voice': args.voice, 'model': 'Kokoro-82M v1.0 (Apache-2.0)', 'clips': {}}

    jobs = []
    for kind, texts in PHRASES.items():
        letters = PER_LETTER.get(kind, [None])
        for L in letters:
            for i, t in enumerate(texts):
                name = f'{kind}_{L}_{i}' if L else f'{kind}_{i}'
                key = f'{kind}_{L}' if L else kind
                jobs.append((name, key, t.replace('{L}', L) if L else t))

    with tempfile.TemporaryDirectory() as tmp:
        for name, key, text in jobs:
            audio, sr = k.create(text, voice=args.voice, speed=args.speed, lang='en-us')
            wav = os.path.join(tmp, name + '.wav')
            sf.write(wav, audio, sr)
            # AAC, mono, 48 kbps: speech stays clear at ~6 KB per second.
            subprocess.run(['afconvert', '-f', 'm4af', '-d', 'aac', '-b', '48000',
                            '-c', '1', wav, os.path.join(OUT, name + '.m4a')],
                           check=True)
            manifest['clips'].setdefault(key, []).append(name + '.m4a')
            if key.startswith(('tip_', 'how_')):
                # The app shows these on screen too; one source for both.
                manifest.setdefault('texts', {}).setdefault(key, text)
    with open(os.path.join(OUT, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)
    size = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))
    print(f'{len(jobs)} clips, {size / 1e6:.1f} MB in {OUT}')


if __name__ == '__main__':
    main()
