# SignAid iOS app

The phone is a client. Inference stays on the Mac, because the pipeline is
MediaPipe hand + pose tracking, a YOLO CNN, a random-forest ensemble and a graph
CNN — porting all of that to CoreML is weeks of work and would cost accuracy.

```
   glasses (Pi + camera)          Mac (app.py)                iPhone (this app)
   ─────────────────────          ────────────                ─────────────────
   capture JPEG  ──Wi-Fi──►  /ingest → models → text  ──Wi-Fi──►  UI + avatar
```

## Building it

Xcode is required and is **not** currently installed on this Mac (only the
Command Line Tools). Install Xcode from the App Store first.

1. Xcode → **File ▸ New ▸ Project ▸ iOS ▸ App**
   - Product Name: `SignAid`
   - Interface: **SwiftUI**, Language: **Swift**
2. Delete the generated `ContentView.swift` and `SignAidApp.swift`.
3. Drag in every `.swift` file from `ios/SignAid/` (tick *Copy items if needed*).
4. In the target's **Info** tab add **App Transport Security Settings** →
   **Allow Arbitrary Loads = YES**. The Mac serves plain HTTP on the LAN, and
   iOS blocks that by default. This is the single most common reason the app
   builds but shows "Can't reach SignAid".
5. Run on the simulator, or on your iPhone (Signing & Capabilities → your Apple
   ID). The phone and the Mac must be on the same Wi-Fi.
6. Open **Settings** in the app and enter the Mac's LAN address.

## Files

| File | Role |
|---|---|
| `API.swift` | Server client: health, detection, autocorrect, avatar fetch |
| `ContentView.swift` | App entry, tab bar, ASL→Text screen |
| `WriteView.swift` | Text→ASL playback and Settings |
| `AvatarCanvas` (`AvatarView.swift`) | Skeleton renderer, mirrors the web version |
| `MJPEGView.swift` | Parses the Mac's MJPEG stream (AVPlayer/WKWebView handle it poorly) |

## Camera options for the glasses

**ESP32-CAM** (common in Arduino car kits) serves MJPEG itself — no Pi needed
for video:

```bash
VIDEO_SOURCE=mjpeg:http://<cam-ip>:81/stream python3 app.py
```

**Raspberry Pi + camera** pushes frames instead:

```bash
# on the Mac
VIDEO_SOURCE=push python3 app.py
# on the Pi
python3 pi_client.py --server http://<mac-ip>:8080 --fps 10
```

## Two hardware facts

- **Bluetooth cannot carry the video.** 640×480 JPEG at 10 fps is ~2–4 Mbps
  measured; Classic Bluetooth SPP peaks near 1 Mbps and BLE is far below.
  Wi-Fi is the only workable link. Bluetooth is fine for control messages.
- **The Pi should not run the models.** Even a Pi 4 would be a slideshow.
  Streaming to the Mac is both faster and simpler.
