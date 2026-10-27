// Generated from camera_pins.h by the script in the README. One entry per
// known ESP32 camera wiring; the firmware tries them in order at boot and
// keeps the first one whose camera answers. That is what lets the same
// firmware run on the ELEGOO robot-car camera, whose wiring ELEGOO does not
// document, and on the XIAO ESP32S3 Sense.
#pragma once

struct CameraBoard { const char *name; int pwdn, reset, xclk, siod, sioc, y9, y8, y7, y6, y5, y4, y3, y2, vsync, href, pclk; };

#if defined(CONFIG_IDF_TARGET_ESP32S3)
static const CameraBoard CAMERA_BOARDS[] = {
  {"XIAO_ESP32S3", -1, -1, 10, 40, 39, 48, 11, 12, 14, 16, 18, 17, 15, 38, 47, 13},
  {"ESP32S3_EYE", -1, -1, 15, 4, 5, 16, 17, 18, 12, 10, 8, 9, 11, 6, 7, 13},
};
#endif
#if !defined(CONFIG_IDF_TARGET_ESP32S3)
static const CameraBoard CAMERA_BOARDS[] = {
  {"M5STACK_WIDE", -1, 15, 27, 22, 23, 19, 36, 18, 39, 5, 34, 35, 32, 25, 26, 21},
  {"WROVER_KIT", -1, -1, 21, 26, 27, 35, 34, 39, 36, 19, 18, 5, 4, 25, 23, 22},
  {"AI_THINKER", 32, -1, 0, 26, 27, 35, 34, 39, 36, 21, 19, 18, 5, 25, 23, 22},
  {"M5STACK_PSRAM", -1, 15, 27, 25, 23, 19, 36, 18, 39, 5, 34, 35, 32, 22, 26, 21},
  {"ESP_EYE", -1, -1, 4, 18, 23, 36, 37, 38, 39, 35, 14, 13, 34, 5, 27, 25},
};
#endif
