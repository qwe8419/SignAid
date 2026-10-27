import Foundation
import SwiftUI
import UIKit

/// Talks to the SignAid server running on the Mac.
///
/// The phone is deliberately a thin client: the pipeline is MediaPipe hand+pose
/// tracking, a YOLO CNN, a random-forest ensemble and a graph CNN. Porting all
/// of that to CoreML is weeks of work and would lose accuracy, so inference
/// stays on the Mac and the app renders results.
@MainActor
final class SignAidClient: ObservableObject {

    @AppStorage("serverHost") var host: String = "192.168.68.60"
    @AppStorage("serverPort") var port: String = "8080"

    @Published var connected = false
    @Published var status: Health?
    @Published var text = ""
    @Published var detected = ""
    @Published var candidates = ""
    @Published var message = ""
    @Published var suggestions: [String] = []
    @Published var autoOn = false
    @Published var autoProgress: Double = 0
    @Published var liveLabel = ""
    @Published var liveConfidence: Double = 0

    /// Set briefly whenever a sign is actually committed, so the UI can flash
    /// and buzz. Without this there is no way to tell a recognised sign from a
    /// sign the app quietly ignored -- the single most confusing thing about
    /// using it.
    @Published var justCommitted: String?
    @Published var commitCount = 0

    private let tap = UIImpactFeedbackGenerator(style: .medium)
    private let nope = UINotificationFeedbackGenerator()

    /// Compares against the previous buffer so a commit is detected wherever it
    /// came from: the auto poller, a manual tap, or a suggestion.
    private func noteCommit(oldText: String, newText: String) {
        guard newText.count > oldText.count, newText.hasPrefix(oldText) else { return }
        let added = String(newText.dropFirst(oldText.count))
            .trimmingCharacters(in: .whitespaces)
        guard !added.isEmpty else { return }
        tap.impactOccurred()
        commitCount += 1
        justCommitted = added
        let token = commitCount
        Task {
            try? await Task.sleep(nanoseconds: 900_000_000)
            if self.commitCount == token { self.justCommitted = nil }
        }
    }

    func signalRejected() {
        nope.notificationOccurred(.warning)
    }

    private var pollTask: Task<Void, Never>?
    /// Stable per-install id so collected bursts can be grouped by session.
    private let sessionID = UUID().uuidString.prefix(8).lowercased()

    var baseURL: URL? { URL(string: "http://\(host):\(port)") }
    var videoURL: URL? { baseURL?.appendingPathComponent("video") }

    struct Health: Decodable {
        let app: String
        let source: String
        let camera_connected: Bool
        let letters: Bool
        let words: Bool
        let avatar_vocabulary: Int
        let auto: Bool
        /// Set by the ESP32 glasses, which stream on their own port.
        let camera_port: Int?
    }

    /// Where the glasses' video comes from: the port they report, else the
    /// same server (the Pi serves /camera alongside everything else).
    var cameraURL: URL? {
        if let p = status?.camera_port { return URL(string: "http://\(host):\(p)/camera") }
        return baseURL?.appendingPathComponent("camera")
    }

    @Published var searching = false

    /// Find the glasses with no typing: their Bonjour name first, then a Pi at
    /// its usual address, then every address an iPhone Personal Hotspot can
    /// hand out (172.20.10.2-14). The first one answering /health wins.
    @discardableResult
    func findGlasses() async -> Bool {
        searching = true
        defer { searching = false }
        var candidates: [(String, String)] = [("signaid-glasses.local", "80")]
        candidates.append((host, port))
        candidates.append(("192.168.68.67", "8080"))
        candidates.append(("192.168.4.1", "80"))      // glasses' own Wi-Fi
        candidates += (2...14).map { ("172.20.10.\($0)", "80") }
        let found: (String, String)? = await withTaskGroup(of: (String, String)?.self) { group in
            for (h, p) in candidates {
                group.addTask {
                    guard let url = URL(string: "http://\(h):\(p)/health") else { return nil }
                    var req = URLRequest(url: url)
                    req.timeoutInterval = 2.5
                    guard let (data, resp) = try? await URLSession.shared.data(for: req),
                          (resp as? HTTPURLResponse)?.statusCode == 200,
                          (try? JSONDecoder().decode(Health.self, from: data)) != nil
                    else { return nil }
                    return (h, p)
                }
            }
            for await r in group { if let r { group.cancelAll(); return r } }
            return nil
        }
        guard let (h, p) = found else { return false }
        host = h
        port = p
        await checkHealth()
        return connected
    }

    struct DetectResult: Decodable {
        let detected: String?
        let candidates: String?
        let word: String
        let message: String?
    }

    struct StateResult: Decodable {
        let word: String
        let auto: Bool
        let live: String?
        let live_conf: Double
        let progress: Double
        let suggestions: [String]
    }

    // MARK: - Networking

    private func request(_ path: String, method: String = "GET",
                         body: [String: Any]? = nil) async throws -> Data {
        guard let base = baseURL else { throw URLError(.badURL) }
        var req = URLRequest(url: base.appendingPathComponent(path))
        req.httpMethod = method
        req.timeoutInterval = 8
        if let body {
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: body)
        }
        let (data, response) = try await URLSession.shared.data(for: req)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }
        return data
    }

    func checkHealth() async {
        do {
            let data = try await request("health")
            status = try JSONDecoder().decode(Health.self, from: data)
            autoOn = status?.auto ?? false
            connected = true
        } catch {
            connected = false
            status = nil
        }
    }

    private func apply(_ r: DetectResult) {
        let previous = text
        detected = r.detected ?? "No sign found"
        candidates = r.candidates ?? ""
        text = r.word
        message = r.message ?? ""
        noteCommit(oldText: previous, newText: r.word)
        if r.detected == "?" || r.detected == nil { signalRejected() }
    }

    func detectLetter() async {
        message = "Detecting — hold the sign steady…"
        if let data = try? await request("detect", method: "POST"),
           let r = try? JSONDecoder().decode(DetectResult.self, from: data) {
            apply(r)
        } else { message = "Could not reach the server" }
    }

    func detectWord() async {
        message = "Recording ~2s — sign the word now…"
        if let data = try? await request("detect_word", method: "POST"),
           let r = try? JSONDecoder().decode(DetectResult.self, from: data) {
            apply(r)
        } else { message = "Could not reach the server" }
    }

    func speak() async {
        _ = try? await request("speak", method: "POST")
        text = ""; detected = ""; candidates = ""; message = "Spoken"
    }

    func deleteLast() async { await simple("delete") }
    func clearText() async { await simple("clear"); detected = ""; candidates = "" }
    func addSpace() async { await simple("space") }

    private func simple(_ path: String) async {
        struct WordOnly: Decodable { let word: String }
        if let data = try? await request(path, method: "POST"),
           let r = try? JSONDecoder().decode(WordOnly.self, from: data) {
            text = r.word
        }
    }

    func autocorrect() async {
        struct Result: Decodable { let word: String; let message: String }
        if let data = try? await request("autocorrect", method: "POST"),
           let r = try? JSONDecoder().decode(Result.self, from: data) {
            text = r.word; message = r.message
        }
    }

    func apply(suggestion: String) async {
        struct WordOnly: Decodable { let word: String }
        if let data = try? await request("apply_suggestion", method: "POST",
                                         body: ["word": suggestion]),
           let r = try? JSONDecoder().decode(WordOnly.self, from: data) {
            text = r.word
        }
    }

    // MARK: - Continuous detection

    func setAuto(_ on: Bool) async {
        autoOn = on
        tap.prepare()
        _ = try? await request("auto", method: "POST", body: ["enabled": on])
        if !on { autoProgress = 0 }
        startPolling()
    }

    /// Poll /state continuously, not only in auto mode. The live readout is the
    /// main way of telling "it sees my hand" from "it sees nothing", which is
    /// just as useful when tapping Detect manually.
    func startPolling() {
        guard pollTask == nil || pollTask?.isCancelled == true else { return }
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                await self?.pollState()
                let fast = await self?.autoOn ?? false
                try? await Task.sleep(nanoseconds: fast ? 150_000_000 : 300_000_000)
            }
        }
    }

    func stopPolling() {
        pollTask?.cancel()
        pollTask = nil
    }

    private func pollState() async {
        guard let data = try? await request("state"),
              let s = try? JSONDecoder().decode(StateResult.self, from: data) else { return }
        let previous = text
        text = s.word
        noteCommit(oldText: previous, newText: s.word)
        autoProgress = s.progress
        liveLabel = s.live ?? ""
        liveConfidence = s.live_conf
        suggestions = s.suggestions
    }

    // MARK: - Data collection

    func uploadSamples(letter: String, burst: Int, variation: String,
                       samples: [[[Double]]]) async -> Bool {
        let body: [String: Any] = [
            "session": sessionID, "burst": burst, "letter": letter,
            "variation": variation, "samples": samples,
        ]
        return (try? await request("collect", method: "POST", body: body)) != nil
    }

    func collectStats() async -> [String: Int]? {
        struct Stats: Decodable { let total: Int; let per_letter: [String: Int] }
        guard let data = try? await request("collect/stats"),
              let s = try? JSONDecoder().decode(Stats.self, from: data) else { return nil }
        return s.per_letter
    }

    // MARK: - Text -> ASL

    struct Segment: Decodable {
        let kind: String
        let label: String
        let letters: [String]?
    }

    struct Translation: Decodable {
        let segments: [Segment]
        let unknown: [String]
        let message: String?
    }

    struct Library: Decodable {
        let letters: [String: [[Double]]]
        let vocabulary: [String]
        let body_edges: [[Int]]
        let hand_edges: [[Int]]
    }

    func translate(_ input: String) async -> Translation? {
        guard let data = try? await request("translate", method: "POST",
                                            body: ["text": input]) else { return nil }
        return try? JSONDecoder().decode(Translation.self, from: data)
    }

    func loadLibrary() async -> Library? {
        guard let data = try? await request("avatar/library") else { return nil }
        return try? JSONDecoder().decode(Library.self, from: data)
    }

    /// One sign's motion, fetched on demand — the full 2000-word vocabulary is
    /// ~40MB, far too much to download up front.
    func wordFrames(_ gloss: String) async -> [[[Double]]]? {
        struct WordResponse: Decodable { let gloss: String; let frames: [[[Double]]] }
        let escaped = gloss.addingPercentEncoding(
            withAllowedCharacters: .urlPathAllowed) ?? gloss
        guard let data = try? await request("avatar/word/\(escaped)") else { return nil }
        return (try? JSONDecoder().decode(WordResponse.self, from: data))?.frames
    }
}
