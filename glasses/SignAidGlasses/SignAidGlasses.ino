// SignAid Glasses — camera firmware for ESP32 camera boards: the Seeed XIAO
// ESP32S3 Sense, and the ESP32 camera module from the ELEGOO Smart Robot Car
// V4 (or any common ESP32-CAM). The camera wiring is detected at boot by
// trying each known layout in camera_boards.h.
//
// The glasses are a camera and nothing else. They stream video over Wi-Fi to
// the iPhone, which runs hand tracking, the letter model and the typing rules
// itself (a Raspberry Pi 4 doing that work overheated and stuttered; the phone
// does it at 30 fps). So this firmware only has to do three things well:
//
//   1. Join the iPhone's Personal Hotspot (works anywhere), or home Wi-Fi.
//   2. Announce itself as signaid-glasses.local so the app finds it without
//      anyone typing an IP address.
//   3. Stream 640x480 JPEG frames as fast as the camera produces them.
//
// HTTP, same shape as the Pi version so the app treats both the same:
//   port 80   /health   JSON status (the app polls this)
//   port 81   /camera   multipart MJPEG, boundary "frame"
// The stream gets its own server and port because an MJPEG response never
// ends, and it would otherwise block /health for as long as it runs.
//
// Build (see glasses/README.md):
//   XIAO ESP32S3 Sense:  --fqbn esp32:esp32:XIAO_ESP32S3:PSRAM=opi
//   ELEGOO / ESP32-CAM:  --fqbn esp32:esp32:esp32wrover
//
// Wi-Fi names and passwords go in secrets.h (copy secrets.example.h). The
// ESP32-S3 radio is 2.4 GHz only: on iPhone, turn on Settings > Personal
// Hotspot > "Maximize Compatibility" or the glasses will not see it.

#include "camera_boards.h"

#include <ESPmDNS.h>
#include <WiFi.h>
#include <WiFiMulti.h>

#include "esp_camera.h"
#include "esp_http_server.h"

#if __has_include("secrets.h")
#include "secrets.h"
#else
#error "Copy secrets.example.h to secrets.h and fill in your Wi-Fi."
#endif

static const char *HOSTNAME = "signaid-glasses";
#if defined(CONFIG_IDF_TARGET_ESP32S3)
static const int LED_PIN = 21;          // XIAO user LED, active LOW
#else
static const int LED_PIN = -1;          // ELEGOO module: no user LED we can drive
#endif
static void led(bool on) { if (LED_PIN >= 0) digitalWrite(LED_PIN, on ? LOW : HIGH); }
static void ledToggle() { if (LED_PIN >= 0) digitalWrite(LED_PIN, !digitalRead(LED_PIN)); }
const char *cameraBoard = "none";

// Camera mounting. The frame holds the module upright; flip these if your
// build mounts it upside down. HMIRROR gives the selfie view the Pi uses
// (the letter model is trained on both hands, so it does not mind either way).
static const bool HMIRROR = true;
static const bool VFLIP = false;

WiFiMulti wifi;
static const char *AP_SSID = "SignAid-Glasses";
static const char *AP_PASSWORD = "signaid123";
bool apMode = false;
httpd_handle_t statusServer = nullptr;
httpd_handle_t streamServer = nullptr;
volatile bool cameraOK = false;
volatile float streamFps = 0;
volatile int streamClients = 0;

// ---------------------------------------------------------------- camera ---

bool initWith(const CameraBoard &b) {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = b.y2;  c.pin_d1 = b.y3;  c.pin_d2 = b.y4;  c.pin_d3 = b.y5;
  c.pin_d4 = b.y6;  c.pin_d5 = b.y7;  c.pin_d6 = b.y8;  c.pin_d7 = b.y9;
  c.pin_xclk = b.xclk;
  c.pin_pclk = b.pclk;
  c.pin_vsync = b.vsync;
  c.pin_href = b.href;
  c.pin_sccb_sda = b.siod;
  c.pin_sccb_scl = b.sioc;
  c.pin_pwdn = b.pwdn;
  c.pin_reset = b.reset;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size = FRAMESIZE_VGA;          // 640x480: the frame the model trains on
  c.jpeg_quality = 12;                   // 10-14: sharp fingers, ~25-40 KB/frame
  c.fb_count = psramFound() ? 2 : 1;     // capture the next while sending this
  c.fb_location = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;      // newest frame, never a queued one
  return esp_camera_init(&c) == ESP_OK;
}

bool startCamera() {
  // Try each known wiring until the camera answers on its I2C bus. A wrong
  // layout simply fails to find the sensor; nothing is driven that could harm
  // the board (none of the layouts touch the PSRAM pins).
  for (const CameraBoard &b : CAMERA_BOARDS) {
    Serial.printf("Trying camera wiring %s... ", b.name);
    if (initWith(b)) {
      Serial.println("found");
      cameraBoard = b.name;
      break;
    }
    Serial.println("no");
    esp_camera_deinit();
    delay(50);
  }
  if (strcmp(cameraBoard, "none") == 0) {
    Serial.println("No camera found. Is the camera ribbon seated? PSRAM enabled?");
    return false;
  }
  sensor_t *s = esp_camera_sensor_get();
  s->set_hmirror(s, HMIRROR ? 1 : 0);
  s->set_vflip(s, VFLIP ? 1 : 0);
  // Hold the frame rate in dim rooms rather than lengthening the exposure:
  // the same choice made on the Pi webcam, where auto-exposure halved fps.
  s->set_exposure_ctrl(s, 1);
  s->set_aec2(s, 1);
  s->set_gain_ctrl(s, 1);
  s->set_gainceiling(s, GAINCEILING_8X);
  Serial.printf("Camera OK, sensor PID 0x%x\n", s->id.PID);
  return true;
}

// ------------------------------------------------------------------ http ---

static esp_err_t healthHandler(httpd_req_t *req) {
  // Field names match the app's Health struct (API.swift) and the Pi server.
  char json[320];
  snprintf(json, sizeof(json),
           "{\"app\":\"SignAid Glasses\",\"board\":\"%s\",\"source\":\"glasses\","
           "\"camera_connected\":%s,\"letters\":false,\"words\":false,"
           "\"avatar_vocabulary\":0,\"auto\":false,\"fps\":%.1f,"
           "\"camera_port\":81,\"clients\":%d,\"rssi\":%d}",
           cameraBoard, cameraOK ? "true" : "false", streamFps, streamClients, WiFi.RSSI());
  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(req, json, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t cameraHandler(httpd_req_t *req) {
  static const char *TYPE = "multipart/x-mixed-replace;boundary=frame";
  static const char *BOUNDARY = "\r\n--frame\r\n";
  static const char *PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";
  char header[64];

  if (httpd_resp_set_type(req, TYPE) != ESP_OK) return ESP_FAIL;
  httpd_resp_set_hdr(req, "Cache-Control", "no-cache");
  streamClients++;
  led(true);                            // LED on while someone is watching

  uint32_t frames = 0, t0 = millis();
  esp_err_t res = ESP_OK;
  while (res == ESP_OK) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) { res = ESP_FAIL; break; }
    size_t n = snprintf(header, sizeof(header), PART, (unsigned)fb->len);
    res = httpd_resp_send_chunk(req, BOUNDARY, strlen(BOUNDARY));
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, header, n);
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
    esp_camera_fb_return(fb);

    frames++;
    uint32_t dt = millis() - t0;
    if (dt >= 2000) { streamFps = frames * 1000.0f / dt; frames = 0; t0 = millis(); }
  }
  if (--streamClients <= 0) { streamClients = 0; streamFps = 0; led(false); }
  return res;
}

void startServers() {
  httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
  cfg.server_port = 80;
  cfg.ctrl_port = 32768;
  httpd_uri_t health = {"/health", HTTP_GET, healthHandler, nullptr};
  if (httpd_start(&statusServer, &cfg) == ESP_OK) httpd_register_uri_handler(statusServer, &health);

  httpd_config_t scfg = HTTPD_DEFAULT_CONFIG();
  scfg.server_port = 81;
  scfg.ctrl_port = 32769;
  scfg.max_open_sockets = 3;
  httpd_uri_t camera = {"/camera", HTTP_GET, cameraHandler, nullptr};
  if (httpd_start(&streamServer, &scfg) == ESP_OK) httpd_register_uri_handler(streamServer, &camera);
}

// ------------------------------------------------------------------ wifi ---

void connectWifi() {
  // Tries every network in secrets.h; the phone's hotspot first, so the
  // glasses work away from home with nothing else to set up.
  for (auto &n : SIGNAID_NETWORKS) wifi.addAP(n.ssid, n.password);
  WiFi.setHostname(HOSTNAME);
  WiFi.setSleep(false);                  // modem sleep stalls the stream
  Serial.print("Joining Wi-Fi");
  uint32_t start = millis();
  while (wifi.run(5000) != WL_CONNECTED) {
    ledToggle();                         // blink while searching
    Serial.print(".");
    if (millis() - start > 25000) {
      // No known network: become one. The phone joins "SignAid-Glasses" and
      // the app finds the glasses at 192.168.4.1. Zero setup, no passwords to
      // type into secrets.h -- good for a first test or a demo table.
      WiFi.mode(WIFI_AP);
      WiFi.softAP(AP_SSID, AP_PASSWORD);
      Serial.printf("\nNo known Wi-Fi. Made network \"%s\" (password %s), "
                    "glasses at %s\n", AP_SSID, AP_PASSWORD,
                    WiFi.softAPIP().toString().c_str());
      apMode = true;
      break;
    }
  }
  if (apMode) {
    if (MDNS.begin(HOSTNAME)) MDNS.addService("signaid", "tcp", 80);
    return;
  }
  led(false);
  Serial.printf("\nConnected to %s as %s\n", WiFi.SSID().c_str(),
                WiFi.localIP().toString().c_str());
  if (MDNS.begin(HOSTNAME)) {
    MDNS.addService("signaid", "tcp", 80);
    MDNS.addServiceTxt("signaid", "tcp", "camera_port", "81");
    MDNS.addService("http", "tcp", 80);
  }
}

void setup() {
  Serial.begin(115200);
  if (LED_PIN >= 0) pinMode(LED_PIN, OUTPUT);
  led(false);
  delay(300);
  Serial.println("\nSignAid Glasses");
  cameraOK = startCamera();
  connectWifi();
  startServers();
  Serial.printf("Ready: http://%s.local/health  stream on :81/camera\n", HOSTNAME);
}

void loop() {
  // Rejoin if the hotspot drops (the phone went out of range or locked).
  if (!apMode && wifi.run(5000) != WL_CONNECTED) {
    Serial.println("Wi-Fi lost, reconnecting");
    connectWifi();
  }
  delay(2000);
}
