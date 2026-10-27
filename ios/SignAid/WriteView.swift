import SwiftUI

struct SettingsView: View {
    @ObservedObject var client: SignAidClient
    @AppStorage("glassesEnabled") private var glassesEnabled = false
    @AppStorage("developerMode") private var developerMode = false
    @AppStorage("seenIntro") private var seenIntro = true
    @AppStorage("practiceBest") private var practiceBest = 0
    @AppStorage("practiceDone") private var practiceDone = 0
    @AppStorage("letterSpeed") private var letterSpeed = "normal"

    var body: some View {
            Form {
                Section {
                    NavigationLink { AboutView() } label: {
                        Label("About, privacy & credits", systemImage: "info.circle")
                    }
                    Button { seenIntro = false } label: {
                        Label("Show the welcome tour again", systemImage: "sparkles")
                    }
                }

                Section {
                    Picker("Letter speed", selection: $letterSpeed) {
                        Text("Quick").tag("quick")
                        Text("Normal").tag("normal")
                        Text("Relaxed").tag("relaxed")
                    }
                } header: {
                    Text("Fingerspelling")
                } footer: {
                    Text("How long to hold each letter before it is typed. Relaxed suits beginners. For a double letter, like the LL in HELLO, slide your hand slightly to the side — the way signers do — or lower it for a moment.")
                }

                Section {
                    LabeledContent("Voice", value: SpeechService.voiceDescription)
                    Button {
                        SpeechService.preview()
                    } label: { Label("Hear it", systemImage: "play.circle") }
                } header: {
                    Text("Speaking voice")
                } footer: {
                    Text(SpeechService.hasNaturalVoice
                         ? "Used for “Say it out loud”. Practice uses its own recorded coach voice."
                         : "For a natural voice instead of the robotic one: open the iPhone Settings app ▸ Accessibility ▸ Read & Speak (Spoken Content) ▸ Voices ▸ English ▸ Zoe or Ava ▸ download the Premium version. SignAid uses it automatically, offline.")
                }

                Section("Practice") {
                    LabeledContent("Letters signed correctly", value: "\(practiceDone)")
                    LabeledContent("Best streak", value: "\(practiceBest)")
                }

                Section {
                    Toggle("SignAid Glasses (prototype)", isOn: $glassesEnabled)
                } footer: {
                    Text("Camera glasses that stream to a computer running the SignAid server. Not needed for the phone app.")
                }

                Section {
                    Toggle("Developer mode", isOn: $developerMode)
                } footer: {
                    Text("Shows model diagnostics and the training-data recorder.")
                }

                if glassesEnabled {
                Section {
                    Button {
                        Task { await client.findGlasses() }
                    } label: {
                        HStack {
                            Label(client.searching ? "Searching…" : "Find my glasses",
                                  systemImage: "eyeglasses")
                            Spacer()
                            if client.connected {
                                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                            }
                        }
                    }
                    .disabled(client.searching)
                } footer: {
                    Text("Turn the glasses on and connect them to this iPhone's Personal Hotspot (with Maximize Compatibility on) or the same Wi-Fi.")
                }
                Section("Server") {
                    // NOT .decimalPad: that made it impossible to type a
                    // hostname like rasberrypi.tail3e7e38.ts.net
                    TextField("Host or Tailscale name", text: $client.host)
                        .keyboardType(.URL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    TextField("Port", text: $client.port)
                        .keyboardType(.numberPad)
                    Button("Test connection") { Task { await client.checkHealth() } }
                }

                Section("Quick connect") {
                    endpoint("Glasses on this Wi-Fi", "192.168.68.67",
                             "Pi on the local network — lowest latency")
                    endpoint("Glasses over Tailscale", "100.100.57.94",
                             "Works from anywhere, both devices on your tailnet")
                    endpoint("Mac on this Wi-Fi", "192.168.68.60",
                             "Full build: words, avatar, autocorrect")
                    endpoint("Mac over Tailscale", "100.98.65.59",
                             "Full build, from anywhere")
                }

                Section("Status") {
                    LabeledContent("Server",
                        value: client.connected ? "Connected" : "Not reachable")
                    if let s = client.status {
                        LabeledContent("Camera source", value: s.source)
                        LabeledContent("Camera",
                            value: s.camera_connected ? "Streaming" : "No frames")
                        LabeledContent("Letter models", value: s.letters ? "Loaded" : "Missing")
                        LabeledContent("Word model", value: s.words ? "Loaded" : "Missing")
                        LabeledContent("Avatar vocabulary", value: "\(s.avatar_vocabulary) signs")
                    }
                }

                Section("Tailscale") {
                    Text("""
                    Tailscale puts your phone, Mac and glasses on one private \
                    network, so SignAid works away from home with no port \
                    forwarding. Install Tailscale on the iPhone and sign in with \
                    the same account, then pick a 100.x address above.

                    Latency is higher than local Wi-Fi, so video will be less \
                    smooth over the internet.
                    """)
                    .font(.caption).foregroundStyle(.secondary)
                }

                Section("Glasses") {
                    Text("""
                    The Pi streams camera frames to the Mac over Wi-Fi, and the \
                    Mac runs the models. Start pi_client.py on the glasses, or \
                    point the server at an ESP32-CAM with \
                    VIDEO_SOURCE=mjpeg:http://<cam-ip>:81/stream.

                    Bluetooth cannot carry the video: it needs 2-4 Mbps and \
                    Bluetooth tops out near 1.
                    """)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
                }
            }
            .navigationTitle("Settings")
    }

    private func endpoint(_ title: String, _ host: String, _ note: String) -> some View {
        Button {
            client.host = host
            client.port = "8080"
            Task { await client.checkHealth() }
        } label: {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).foregroundStyle(.primary)
                    Text(note).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                if client.host == host {
                    Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                }
            }
        }
    }
}
