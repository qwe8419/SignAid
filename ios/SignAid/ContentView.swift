import SwiftUI

@main
struct SignAidApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

struct ContentView: View {
    @StateObject private var client = SignAidClient()
    @AppStorage("seenIntro") private var seenIntro = false
    /// The glasses are a prototype that needs a Mac on the same network. Off by
    /// default so a first-time user only ever sees the part that works anywhere.
    @AppStorage("glassesEnabled") private var glassesEnabled = false
    @AppStorage("usePhoneCamera") private var usePhoneCamera = true

    /// Only talk to the server when the glasses are actually in use. Polling a
    /// LAN address from phone-only mode cost battery and raised iOS's
    /// local-network permission prompt on first launch for no reason.
    private var wantsServer: Bool { glassesEnabled && !usePhoneCamera }

    var body: some View {
        TranslateView(client: client)
            .tint(.green)
            .preferredColorScheme(.dark)
            .task(id: wantsServer) {
                guard wantsServer else { client.stopPolling(); return }
                await client.checkHealth()
                client.startPolling()          // live readout from the moment it connects
                while !Task.isCancelled {
                    try? await Task.sleep(nanoseconds: 4_000_000_000)
                    await client.checkHealth()
                }
            }
            .fullScreenCover(isPresented: Binding(get: { !seenIntro },
                                                  set: { seenIntro = !$0 })) {
                IntroView { seenIntro = true }
            }
    }
}

struct ActionButton: View {
    let title: String
    let color: Color
    let action: () async -> Void

    init(_ title: String, _ color: Color, action: @escaping () async -> Void) {
        self.title = title; self.color = color; self.action = action
    }

    var body: some View {
        Button { Task { await action() } } label: {
            Text(title).frame(maxWidth: .infinity).padding(.vertical, 6)
        }
        .buttonStyle(.borderedProminent)
        .tint(color)
    }
}

struct DisconnectedCard: View {
    @ObservedObject var client: SignAidClient

    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: "wifi.exclamationmark").font(.largeTitle)
            Text("Can't reach SignAid").font(.headline)
            Text("Start `python3 app.py` on the Mac and check the address in Settings.")
                .font(.caption).multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
            Button("Retry") { Task { await client.checkHealth() } }
                .buttonStyle(.borderedProminent)
        }
        .frame(maxWidth: .infinity, minHeight: 200)
        .background(Color(white: 0.12))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}
