# SignAid

Two-way ASL ↔ English conversation on one iPhone, entirely on-device.

- **Sign → speech.** Fingerspell to the camera; SignAid types each letter,
  suggests the word, and speaks it aloud.
- **Speech → sign.** The hearing person talks or types; known words play a
  recording of a Deaf signer, everything else is fingerspelled.
- **Practice.** The same recogniser as a tutor: it shows a letter and tells you
  when you've signed it.

No account, no network, no video stored.

## How recognition works

```
camera frame ─► MediaPipe Hand Landmarker (21 points)
             ─► HandFeatures.toModelFrame   (sensor frame → training frame)
             ─► 300 geometric features      (features.py / HandFeatures.swift)
             ─► 3-model MLP ensemble        (letter_net_3d.bin, ~1 MB)
             ─► 9-frame confidence vote     (commit only when stable)
             ─► text ─► autocorrect ─► speech
```

## Results

All numbers are on data the model was **not** trained on.

| Test | Before | Now |
|---|---|---|
| Author's phone, per frame | 0.579 | **0.965** |
| Author's phone, letters committed by the app | 0.156 | **0.906** |
| 10 new signers (ASL-HG), leave-signers-out, 5 folds | 0.790 | **0.937 ± 0.021** |

"Before" for new signers is a model trained only on the author's hand; "now"
adds the other ASL-HG signers, never the ones being tested.

The phone jump came from a bug, not a bigger model. MediaPipe returns phone
landmarks in the camera sensor's landscape frame, a quarter-turn from
upright. Rotation is normalised out of 298 of the 300 features, but the last
two encode which way the hand points: the only thing separating H from U, G
from Q and P from K. Every upright two-finger letter read as H. See
`train_robust.py` for the measurements.

Evaluation is always split by capture burst or by person, never by frame:
frames within a burst are near-duplicates, and a random split once reported
100% for a model that scored 56% under rotation.

## Repository

| Path | What |
|---|---|
| `ios/` | The app (Swift, SwiftUI). `pod install`, then open `SignAid.xcworkspace`. |
| `train_robust.py` | Trains and exports the shipped letter model; prints the evaluation table. |
| `features.py` | Feature extractor; `ios/SignAid/HandFeatures.swift` must match it. |
| `extract_image_letters.py` | Photos (folder or zip) → MediaPipe landmarks. |
| `test_shipped.py` | Runs the exact weights in the app over recorded phone bursts. |
| `authoring/fetch_popsign.py` | Builds `sign_videos/` from PopSign (CC BY 4.0). |
| `authoring/studio_backdrop.swift` | Cuts the signer out (Apple person segmentation) onto one studio backdrop, 480p. |
| `app.py`, `pi_standalone.py`, `hardware/` | SignAid Glasses prototype (Pi camera + Mac server). |
| `RESEARCH_DESIGN.md` | Study design for the confusion-type research question. |

### Retraining

```bash
python3 extract_image_letters.py ~/asl_project/my_asl_dataset image_letters/my_asl.json
python3 extract_image_letters.py ASL_Raw_Images.zip image_letters/asl_hg.json
python3 train_robust.py            # evaluate, then export to ios/SignAid/
```

## Credits

- **PopSign ASL v1.0**, Starner et al., Georgia Tech and RIT/NTID. Word-sign
  videos by a Deaf adult signer. CC BY 4.0. One clip per sign, trimmed,
  placed on a plain studio backdrop and downscaled.
- **ASL-HG**, American Sign Language Hand Gesture Image Dataset, Mendeley Data,
  doi:10.17632/j4y5w2c8w9.1. CC BY 4.0. Hand landmarks extracted for training.
- **Wikimedia Commons** "Sign language A–Z" drawings. Public domain.
- **MediaPipe** Hand Landmarker, Google. Apache 2.0.
- **Kokoro-82M** (hexgrad), Apache 2.0. Practice-coach voice, pre-recorded
  by `authoring/make_coach_voice.py` into `coach_voice/`.

## Signer videos
The 246 PopSign ASL clips (55 MB, CC BY 4.0) are not in this repo to keep it small. Regenerate them with `python3 authoring/fetch_popsign.py --signer 4a.8902 --strict` then `swift authoring/studio_backdrop.swift sign_videos_raw sign_videos`.
