import AVFoundation
import SwiftUI
import UIKit

/// Fingerspelling practice: the app shows a letter, you sign it, it tells you
/// when it sees it.
///
/// Runs on the same recogniser and camera as translation, with auto-commit off,
/// so nothing is typed while practising. The bar for "got it" is the recogniser
/// reading the target at >= `passConfidence` for `holdFrames` frames out of a
/// rolling count, which is about half a second of a clean handshape: long
/// enough that a sign passing through on the way to another does not count.
///
/// J and Z are left out. Both are defined by movement, and the letter model
/// reads single frames.
///
/// Audio coach: practice without looking. It says the letter to sign, a strong
/// haptic and "Yes!" confirm it, and spoken hints cover a wrong letter or a
/// missing hand. Turning the phone around puts the BACK camera on you, so the
/// coach switches to it; the screen is kept awake because nobody is touching it.
struct PracticePanel: View {
    @ObservedObject var onDevice: OnDeviceRecognizer

    static let letters = "ABCDEFGHIKLMNOPQRSTUVWXY".map(String.init)
    private let passConfidence = 0.55
    private let holdFrames = 10

    @State private var target = PracticePanel.letters.randomElement()!
    @State private var hold = 0
    @State private var streak = 0
    @State private var passed = false
    @AppStorage("practiceBest") private var best = 0
    @AppStorage("practiceDone") private var done = 0
    @AppStorage("practiceAudio") private var audioCoach = false
    private let haptic = UINotificationFeedbackGenerator()
    private let tick = UIImpactFeedbackGenerator(style: .light)
    @State private var voice = PracticeVoice()
    // Throttles for spoken hints, so the coach nudges rather than nags.
    @State private var wrongSince: Date?
    @State private var lastHint = Date.distantPast
    @State private var noHandSince: Date?
    @State private var tickedThisLetter = false
    @State private var letterStart = Date()
    @State private var encouraged = false
    // Hand-shape coach: a tip appears once one has been the most common
    // diagnosis over the last ~half second, so it does not flicker.
    @State private var coach = HandCoach()
    @State private var recentTips: [String?] = []
    @State private var shownTip: String?
    @State private var lastSpokenTip: String?
    @State private var showHowTo = false

    var body: some View {
        VStack(spacing: 12) {
            HStack(alignment: .center, spacing: 14) {
                LetterPhotoView(letter: target)
                    .frame(width: 128, height: 128)
                    .clipShape(RoundedRectangle(cornerRadius: 14))
                    .overlay(RoundedRectangle(cornerRadius: 14)
                        .stroke(passed ? .green : .clear, lineWidth: 3))
                VStack(alignment: .leading, spacing: 6) {
                    Text("Sign this letter").font(.caption).foregroundStyle(.secondary)
                    Text(target)
                        .font(.system(size: 56, weight: .bold, design: .rounded))
                        .foregroundStyle(passed ? .green : .white)
                        .contentTransition(.numericText())
                    Text(hint).font(.callout).foregroundStyle(hintColor)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
            }

            ProgressView(value: Double(min(hold, holdFrames)), total: Double(holdFrames))
                .tint(passed ? .green : .orange)

            if !passed, let tip = shownTip, let text = coach.text(tip) {
                Label(text, systemImage: "hand.point.up.left.fill")
                    .font(.callout.weight(.semibold)).foregroundStyle(.yellow)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(10)
                    .background(Color.yellow.opacity(0.12))
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                    .transition(.opacity)
            }
            if showHowTo, let how = coach.text("how_\(target)") {
                // The spoken version opens with "Here's how to sign W."; on
                // screen the big letter already says that.
                Text(how.replacingOccurrences(of: "Here's how to sign \(target). ", with: "")).font(.callout).foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            HStack {
                Label("\(streak) in a row", systemImage: "flame.fill")
                    .foregroundStyle(streak > 0 ? .orange : .secondary)
                Spacer()
                Label("Best \(best)", systemImage: "trophy.fill")
                    .foregroundStyle(.secondary)
            }
            .font(.callout).lineLimit(1)

            HStack(spacing: 8) {
                controlButton(audioCoach ? "Voice on" : "Voice off",
                              audioCoach ? "speaker.wave.2.fill" : "speaker.slash.fill",
                              tint: audioCoach ? .green : .gray) {
                    audioCoach.toggle()
                    applyAudioCoach(announce: true)
                }
                controlButton("How?", "questionmark.circle.fill", tint: .yellow) {
                    showHowTo = true
                    if audioCoach { say(["how_\(target)"]) }
                }
                controlButton("Skip", "forward.fill", tint: .gray) { advance() }
            }
        }
        .padding(14)
        .background(Color(white: 0.1))
        .clipShape(RoundedRectangle(cornerRadius: 14))
        .animation(.spring(response: 0.3, dampingFraction: 0.7), value: passed)
        .animation(.easeInOut(duration: 0.2), value: shownTip)
        .animation(.easeInOut(duration: 0.2), value: target)
        .onAppear {
            onDevice.autoCommit = false
            // Practice is hands-busy: keep the screen (and camera) from locking.
            UIApplication.shared.isIdleTimerDisabled = true
            applyAudioCoach(announce: audioCoach)
        }
        .onDisappear {
            onDevice.autoCommit = true
            UIApplication.shared.isIdleTimerDisabled = false
            voice.stop()
        }
        .task {
            // Once a second: remind about a missing hand when nothing is visible.
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                remindIfNoHand()
                repeatTipIfStillNeeded()
            }
        }
        // Speak a tip the moment it appears on screen.
        .onChange(of: shownTip) { _, tip in
            guard audioCoach, let tip, !passed,
                  Date().timeIntervalSince(lastHint) > 1.5 || tip != lastSpokenTip else { return }
            lastSpokenTip = tip
            say([tip])
        }
        // Fires on every recognised frame: confidence changes frame to frame.
        .onChange(of: onDevice.liveConfidence) { _, _ in check() }
        .onChange(of: onDevice.handVisible) { _, visible in if !visible { hold = 0 } }
    }

    /// Same shape as Translate's Space / Delete / Fix / Clear buttons.
    private func controlButton(_ title: String, _ icon: String, tint: Color,
                               action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 3) {
                Image(systemName: icon).font(.system(size: 17))
                Text(title).font(.caption).lineLimit(1)
            }
            .frame(maxWidth: .infinity).padding(.vertical, 6)
        }
        .buttonStyle(.bordered).tint(tint)
    }

    /// Every spoken line goes through here, so the repeat timers know when
    /// the coach last talked.
    private func say(_ lines: [String]) {
        lastHint = Date()
        voice.play(lines)
    }

    private var hint: String {
        if passed { return "Nice! That's \(target)." }
        if !onDevice.handVisible { return "Hold your hand up to the camera" }
        if onDevice.liveLabel == target { return "That's it — hold it…" }
        if onDevice.liveLabel.isEmpty { return "Looking…" }
        return "I'm seeing \(onDevice.liveLabel). Compare with the picture."
    }

    private var hintColor: Color {
        if passed || onDevice.liveLabel == target { return .green }
        return .secondary
    }

    private func check() {
        guard !passed else { return }
        noHandSince = nil
        if onDevice.liveLabel == target && onDevice.liveConfidence >= passConfidence {
            hold += 1
            wrongSince = nil
            if audioCoach && !tickedThisLetter && hold >= 3 {
                tick.impactOccurred()            // "you're on it, keep holding"
                tickedThisLetter = true
            }
        } else {
            hold = max(0, hold - 2)
            updateCoachTip()
            hintIfStuck()
        }
        if hold >= holdFrames { succeed() }
    }

    /// The same wrong letter held for 2.5 s gets one spoken hint, at most
    /// every 5 s.
    private func hintIfStuck() {
        guard audioCoach, !onDevice.liveLabel.isEmpty else { return }
        let now = Date()
        if wrongSince == nil { wrongSince = now }
        guard let since = wrongSince, now.timeIntervalSince(since) > 2.5,
              now.timeIntervalSince(lastHint) > 5 else { return }
        lastHint = now
        wrongSince = nil
        haptic.notificationOccurred(.warning)
        // After a while on one letter, lead with encouragement rather than
        // another correction.
        if !encouraged && now.timeIntervalSince(letterStart) > 10 {
            encouraged = true
            say(["encourage", "try_\(target)"])
        } else if shownTip == nil {
            // No specific fix to suggest: at least say what it is seeing.
            say(["looks_like_\(onDevice.liveLabel)", "try_\(target)"])
        }
    }

    /// Still showing the same tip after 7 s of silence: say it again, with a
    /// word of encouragement first now and then.
    private func repeatTipIfStillNeeded() {
        guard audioCoach, !passed, let tip = shownTip, onDevice.handVisible,
              Date().timeIntervalSince(lastHint) > 7 else { return }
        lastSpokenTip = tip
        say(Bool.random() ? ["encourage", tip] : [tip])
    }

    private func remindIfNoHand() {
        guard audioCoach, !passed else { return }
        if onDevice.handVisible { noHandSince = nil; return }
        let now = Date()
        if noHandSince == nil { noHandSince = now }
        guard let since = noHandSince, now.timeIntervalSince(since) > 4,
              now.timeIntervalSince(lastHint) > 8 else { return }
        lastHint = now
        say(["no_hand", "try_\(target)"])
    }

    private func applyAudioCoach(announce: Bool) {
        guard audioCoach else { voice.stop(); return }
        if announce { say(["audio_on"]) }
    }

    private func succeed() {
        passed = true
        streak += 1
        done += 1
        best = max(best, streak)
        haptic.notificationOccurred(.success)
        if audioCoach {
            // A second, heavier buzz so success is unmistakable by feel alone.
            UIImpactFeedbackGenerator(style: .heavy).impactOccurred(intensity: 1)
        }
        let next = nextLetter()
        if audioCoach {
            var lines = ["praise"]
            if [5, 10, 20].contains(streak) { lines.append("streak_\(streak)") }
            lines.append("next_\(next)")
            say(lines)
        }
        Task {
            try? await Task.sleep(nanoseconds: 900_000_000)
            advance(to: next, keepStreak: true)
        }
    }

    private func nextLetter() -> String {
        var next = target
        while next == target { next = Self.letters.randomElement()! }
        return next
    }

    private func advance(to chosen: String? = nil, keepStreak: Bool = false) {
        if !keepStreak { streak = 0 }
        let next = chosen ?? nextLetter()
        if audioCoach && !keepStreak { say(["skip", "next_\(next)"]) }
        target = next
        hold = 0
        passed = false
        tickedThisLetter = false
        wrongSince = nil
        letterStart = Date()
        encouraged = false
        recentTips.removeAll()
        shownTip = nil
        lastSpokenTip = nil
        showHowTo = false
    }

    /// Diagnose the current hand against the target letter. Shown only after
    /// two seconds on the letter, so a quick correct sign never sees a tip.
    private func updateCoachTip() {
        guard Date().timeIntervalSince(letterStart) > 2,
              let pts = onDevice.lastModelPoints else { return }
        recentTips.append(coach.tip(target: target, points: pts))
        if recentTips.count > 15 { recentTips.removeFirst(recentTips.count - 15) }
        let counts = Dictionary(grouping: recentTips.compactMap { $0 }, by: { $0 })
        if let (tip, hits) = counts.max(by: { $0.value.count < $1.value.count }),
           hits.count * 10 >= recentTips.count * 6 {
            shownTip = tip
        } else if recentTips.count >= 10 && recentTips.filter({ $0 == nil }).count > recentTips.count / 2 {
            shownTip = nil              // mostly right now; take the tip away
        }
    }
}

/// The audio coach's voice: short clips recorded once with a natural neural
/// voice (authoring/make_coach_voice.py, Kokoro, Apache-2.0) and bundled in
/// coach_voice/, so it stays offline. play() takes clip kinds ("praise",
/// "next_K"), picks one recorded wording of each at random, and plays them back
/// to back. Plays with the silent switch on, and ducks music rather than
/// stopping it.
@MainActor
final class PracticeVoice: NSObject, AVAudioPlayerDelegate {
    private var clips: [String: [String]] = [:]
    private let root = Bundle.main.url(forResource: "coach_voice", withExtension: nil)
    private var queue: [URL] = []
    private var player: AVAudioPlayer?
    private var lastPick: [String: String] = [:]

    override init() {
        super.init()
        if let root,
           let data = try? Data(contentsOf: root.appendingPathComponent("manifest.json")),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            clips = (obj["clips"] as? [String: [String]]) ?? [:]
        }
    }

    func play(_ kinds: [String]) {
        guard let root else { return }
        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
        try? session.setActive(true)
        stop()
        queue = kinds.compactMap { kind in
            guard var options = clips[kind], !options.isEmpty else { return nil }
            // Avoid saying the exact same wording twice in a row.
            if options.count > 1, let last = lastPick[kind] { options.removeAll { $0 == last } }
            let pick = options.randomElement()!
            lastPick[kind] = pick
            return root.appendingPathComponent(pick)
        }
        playNext()
    }

    private func playNext() {
        guard !queue.isEmpty else { player = nil; return }
        let url = queue.removeFirst()
        player = try? AVAudioPlayer(contentsOf: url)
        player?.delegate = self
        player?.play()
    }

    nonisolated func audioPlayerDidFinishPlaying(_ p: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in self.playNext() }
    }

    func stop() {
        queue.removeAll()
        player?.stop()
        player = nil
    }
}


/// The hand-shape coach: says WHAT to change, not just that the letter is
/// wrong. Each letter has the few measurements that define it (finger bends,
/// which way the hand points, finger spread or crossing, thumb placement), with
/// normal ranges learned from 33k real hands (build_coach.py ->
/// letter_coach.json). The check missed by the most becomes the tip. Measured
/// on held-out confusions, 97% of tips name the real difference, and 98% of
/// correctly signed frames get no tip.
final class HandCoach {
    private struct Check { let feature: String; let lo, hi, scale, median: Double }
    private var letters: [String: [Check]] = [:]
    private var texts: [String: String] = [:]
    private let names = ["curl_thumb", "curl_index", "curl_middle", "curl_ring",
                         "curl_pinky", "point_down", "spread_im", "cross_im",
                         "thumb_index", "thumb_middle", "thumb_out"]
    private let margin = 0.5

    init() {
        if let url = Bundle.main.url(forResource: "letter_coach", withExtension: "json"),
           let data = try? Data(contentsOf: url),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let table = obj["letters"] as? [String: [[String: Any]]] {
            for (letter, checks) in table {
                letters[letter] = checks.compactMap { c in
                    guard let f = c["feature"] as? String, let lo = c["lo"] as? Double,
                          let hi = c["hi"] as? Double, let sc = c["scale"] as? Double,
                          let med = c["median"] as? Double else { return nil }
                    return Check(feature: f, lo: lo, hi: hi, scale: sc, median: med)
                }
            }
        }
        if let root = Bundle.main.url(forResource: "coach_voice", withExtension: nil),
           let data = try? Data(contentsOf: root.appendingPathComponent("manifest.json")),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            texts = (obj["texts"] as? [String: String]) ?? [:]
        }
    }

    func text(_ key: String) -> String? { texts[key] }

    /// -> tip key (e.g. "tip_point_down"), or nil when the hand looks right.
    func tip(target: String, points: [[Double]]) -> String? {
        guard points.count == 21, let checks = letters[target] else { return nil }
        let f = HandFeatures.coachFeatures(points)
        var worst: (Check, Int)?
        var best = margin
        for c in checks {
            guard let i = names.firstIndex(of: c.feature) else { continue }
            let v = f[i]
            let miss = v > c.hi ? (v - c.hi) / c.scale : v < c.lo ? (c.lo - v) / c.scale : 0
            if miss > best { best = miss; worst = (c, v > c.hi ? 1 : -1) }
        }
        guard let (c, dir) = worst else { return nil }
        switch c.feature {
        case let f where f.hasPrefix("curl_"):
            let finger = f.dropFirst(5)
            return "tip_curl_\(finger)_" + (dir > 0 ? "straighten" : "fold")
        case "point_down":
            if dir < 0 {      // pointing further up than this letter allows
                return (c.median > 0.4 || target == "P") ? "tip_point_down" : "tip_point_side"
            }
            if target == "P" { return "tip_point_forward" }
            return c.median < -0.5 ? "tip_point_up" : "tip_point_side"
        case "spread_im": return dir > 0 ? "tip_spread_together" : "tip_spread_apart"
        case "cross_im": return dir > 0 ? "tip_cross_do" : "tip_cross_undo"
        case "thumb_index": return dir > 0 ? "tip_thumbindex_touch" : "tip_thumbindex_away"
        case "thumb_middle": return dir > 0 ? "tip_thumbmiddle_touch" : "tip_thumbmiddle_away"
        case "thumb_out": return dir > 0 ? "tip_thumbout_in" : "tip_thumbout_out"
        default: return nil
        }
    }
}
