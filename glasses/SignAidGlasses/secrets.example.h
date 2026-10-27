// Copy this file to secrets.h and fill in your networks. secrets.h is
// git-ignored so passwords never end up in the repository.
//
// Order matters: the first network found is used. Put the iPhone hotspot first
// so the glasses work anywhere. On the iPhone, turn on Settings > Personal
// Hotspot > "Maximize Compatibility" (the glasses can only use 2.4 GHz).
#pragma once

struct SignAidNetwork { const char *ssid; const char *password; };

static const SignAidNetwork SIGNAID_NETWORKS[] = {
  {"Your iPhone's hotspot name", "hotspot-password"},
  {"Home Wi-Fi name", "home-password"},
};
