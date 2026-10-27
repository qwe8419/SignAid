# SignAid Glasses

The glasses are a **camera and nothing else**. They stream video over Wi-Fi to
the iPhone in your pocket, which runs hand tracking, the letter model and the
typing rules: the same code as the phone's own camera, so the two behave
identically. A battery pack in the other pocket powers the glasses over a
cable, the way the Vision Pro's battery does, so the frame stays light.

```
 glasses (camera + Wi-Fi)  ──Wi-Fi──►  iPhone (all AI, text, speech, videos)
        │
     USB-C cable
        │
 pocket battery pack
```

Why not a computer in the glasses or the pocket: a Raspberry Pi 4 doing hand
tracking ran at 82 °C, throttled itself and stuttered at ~8 fps. The iPhone does
the same work at 30 fps without noticing. Bluetooth cannot carry the video
(needs ~3–4 Mbit/s); Wi-Fi can.

## Parts

| Part | Why | Approx. price |
|---|---|---|
| **Seeed Studio XIAO ESP32S3 Sense** | Camera + Wi-Fi on a 21 × 17.8 mm board; the Sense add-on carries the camera. Check the listing for the OV3660 sensor (newer) — OV2640 also works. | ~$14 |
| **USB-C power bank, 5000 mAh, small** | The pocket pack. Pick one that does *not* auto-shut-off at low current (or has a "low-current"/"always-on" mode); the glasses draw ~0.25 A. | ~$15–25 |
| **Thin braided USB-C cable, 1 m, right-angle on one end** | Runs from the pack, behind the ear, along the temple. Right-angle end keeps it flat against the frame. | ~$8 |
| **Small stick-on heatsink, ~10 × 10 mm** | The ESP32-S3 gets warm streaming; the frame needs this plus a vent slot. | ~$5 for a pack |
| *Optional:* **3.7 V LiPo, 300–500 mAh, JST 1.25 mm or bare leads** | Cable-free option: the XIAO charges a LiPo on its BAT pads over USB-C. ~1–1.5 h of streaming. | ~$8 |
| *Optional:* **Wide-angle OV2640/OV3660 module (120–160°)** | Only if the *signer* wears the glasses — the stock lens is too narrow to see your own hands. Not needed if the hearing person wears them. | ~$8 |
| 3D-printed frame + battery case | PLA or PETG; a school/library printer is fine. Model: `hardware/`. | filament |

Total: about **$40–60**.

## Flashing the firmware

1. `cp glasses/SignAidGlasses/secrets.example.h glasses/SignAidGlasses/secrets.h`
   and put your iPhone hotspot name/password first, home Wi-Fi second.
2. On the iPhone: **Settings ▸ Personal Hotspot ▸ Maximize Compatibility** on
   (the ESP32 only speaks 2.4 GHz).
3. Plug the XIAO into the Mac with USB-C, then:

```bash
arduino-cli compile --fqbn esp32:esp32:XIAO_ESP32S3:PSRAM=opi glasses/SignAidGlasses
arduino-cli upload  --fqbn esp32:esp32:XIAO_ESP32S3:PSRAM=opi -p /dev/cu.usbmodem* glasses/SignAidGlasses
arduino-cli monitor -p /dev/cu.usbmodem* -c baudrate=115200
```

The monitor prints `Ready: http://signaid-glasses.local/health`. The orange LED
blinks while joining Wi-Fi and stays on while the phone is watching.

## Connecting from the app

**Settings ▸ SignAid Glasses** on ▸ **Find my glasses**. The app tries
`signaid-glasses.local`, then the Pi, then every iPhone-hotspot address, and
uses whichever answers. Then pick **Glasses** on the ASL screen.

## Endpoints

| Port | Path | What |
|---|---|---|
| 80 | `/health` | JSON status, including `camera_port` and stream `fps` |
| 81 | `/camera` | MJPEG, 640 × 480, boundary `frame` |

The Raspberry Pi version (`pi_standalone.py`, camera-only by default) serves the
same `/health` and `/camera` on port 8080, so the app treats both the same.
