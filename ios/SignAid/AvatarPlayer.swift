import SwiftUI

/// The playback half of the conversation: English in, ASL out.
///
/// Everything shown here is a recording of a real person. Words play as video
/// of a signer; anything with no clip is fingerspelled from photographs of a
/// real hand. There is no avatar and no reconstruction.
///
/// There used to be one — a rigged 3-D character driven from landmark data —
/// and it was removed rather than improved. Landmarks describe where joints
/// are, not what a body looks like doing it, so posing a rig from them lands in
/// the uncanny valley by construction: every fix for one sign was an
/// approximation that broke another. A recording has none of that problem
/// because there is nothing left to infer.
struct AvatarPlayer: View {
    @ObservedObject var client: SignAidClient
    @ObservedObject var offline: OfflineAvatar
    @Binding var text: String

    @State private var videoClip: URL?
    @State private var letterPhoto: String?
    @State private var nowSigning = ""
    @State private var gloss = ""
    @State private var speed: Double = 1.0
    @State private var playTask: Task<Void, Never>?
    /// The last sentence signed, so the Deaf person can watch it again.
    @State private var lastPlayed = ""

    private var idle: Bool { videoClip == nil && letterPhoto == nil }

    var body: some View {
        ZStack(alignment: .bottom) {
            Group {
                if let clip = videoClip {
                    SignVideoView(clip: clip, speed: Float(speed))
                } else if let letter = letterPhoto {
                    LetterPhotoView(letter: letter)
                } else {
                    Color(white: 0.06)
                }
            }
            .overlay(alignment: .center) {
                if idle {
                    VStack(spacing: 8) {
                        Image(systemName: "hands.and.sparkles")
                            .font(.system(size: 40)).foregroundStyle(.green.opacity(0.8))
                        Text(offline.ready ? "Speak or type, then tap Sign it" : "Loading signs…")
                            .font(.callout).foregroundStyle(.secondary)
                    }
                }
            }
            .overlay(alignment: .topLeading) {
                if !idle {
                    Text("ASL")
                        .font(.caption.weight(.bold)).foregroundStyle(.black)
                        .padding(.horizontal, 8).padding(.vertical, 3)
                        .background(Capsule().fill(.green))
                        .padding(10)
                }
            }
            .overlay(alignment: .topTrailing) {
                if videoClip != nil {
                    Text("Deaf signer · PopSign")
                        .font(.caption2).foregroundStyle(.white.opacity(0.75))
                        .padding(.horizontal, 7).padding(.vertical, 3)
                        .background(Capsule().fill(.black.opacity(0.45)))
                        .padding(10)
                }
            }

            VStack(spacing: 4) {
                if !nowSigning.isEmpty {
                    Text(nowSigning)
                        .font(.system(size: 26, weight: .bold, design: .rounded))
                        .foregroundStyle(.white)
                        .lineLimit(1).minimumScaleFactor(0.6)
                }
                if !gloss.isEmpty {
                    Text(gloss)
                        .font(.caption).foregroundStyle(.orange)
                        .lineLimit(2).multilineTextAlignment(.center)
                }
                HStack {
                    Image(systemName: "tortoise.fill").font(.caption2)
                    Slider(value: $speed, in: 0.4...2.0, step: 0.1)
                    Image(systemName: "hare.fill").font(.caption2)
                    if playTask == nil && !lastPlayed.isEmpty {
                        Button { play(lastPlayed) } label: {
                            Image(systemName: "arrow.counterclockwise.circle.fill")
                        }
                        .tint(.green)
                    }
                    if playTask != nil {
                        Button {
                            playTask?.cancel()
                            playTask = nil
                            clear()
                        } label: { Image(systemName: "stop.circle.fill") }
                            .tint(.red)
                    }
                }
                .foregroundStyle(.secondary)
            }
            .padding(10)
            .background(LinearGradient(colors: [.clear, .black.opacity(0.85)],
                                       startPoint: .top, endPoint: .bottom))
        }
        .task { offline.loadIndex() }
        .onReceive(NotificationCenter.default.publisher(for: .signNow)) { note in
            if let t = note.object as? String { play(t) }
        }
    }

    private func clear() {
        nowSigning = ""
        videoClip = nil
        letterPhoto = nil
    }

    private func play(_ input: String) {
        lastPlayed = input
        playTask?.cancel()
        playTask = Task {
            // Entirely on-device: no server, no network.
            let (segments, _) = offline.plan(input)
            gloss = segments.map {
                $0.kind == "word" ? $0.label.uppercased()
                                  : "-" + $0.letters.joined(separator: "-") + "-"
            }.joined(separator: "   ")

            for segment in segments {
                if Task.isCancelled { break }
                if segment.kind == "word" {
                    await playWord(segment.gloss, caption: segment.label)
                } else {
                    await playLetters(segment)
                }
            }
            if !Task.isCancelled { clear() }
            playTask = nil
        }
    }

    private func playWord(_ gloss: String, caption: String) async {
        guard let url = SignVideos.url(for: gloss) else {
            // No clip: spell it, which is what a signer does for a word they
            // have no sign for.
            await playLetters(OfflineAvatar.Segment(
                kind: "fingerspell", label: caption,
                letters: caption.uppercased().map(String.init)))
            return
        }
        // Caption shows the English word; the clip is whatever sign carries it.
        nowSigning = caption.uppercased()
            + (gloss == caption.lowercased() ? "" : "  (\(gloss.uppercased()))")
        letterPhoto = nil
        videoClip = url
        let seconds = await SignVideos.duration(of: url)
        try? await Task.sleep(nanoseconds: UInt64(seconds / speed * 1_000_000_000))
        guard !Task.isCancelled else { return }
        // A beat between signs, so two words do not run together.
        try? await Task.sleep(nanoseconds: UInt64(140_000_000 / speed))
    }

    private func playLetters(_ segment: OfflineAvatar.Segment) async {
        videoClip = nil
        for ch in segment.letters {
            if Task.isCancelled { return }
            guard LetterPhotos.has(ch) else { continue }
            nowSigning = "\(segment.label.uppercased())  —  \(ch)"
            letterPhoto = ch
            try? await Task.sleep(nanoseconds: UInt64(520_000_000 / speed))
        }
        try? await Task.sleep(nanoseconds: UInt64(200_000_000 / speed))
    }
}
