import AVFoundation
import SwiftUI

/// Plays one sign as video of a real signer.
///
/// The avatar reconstructs signing; this shows it. A recording of a person
/// signing is correct by construction — no depth to recover, no wrist to clamp,
/// no handshape to infer — which is the whole reason to prefer it.
///
/// One clip at a time rather than a queue, because a sentence is not all video:
/// words with no clip still fingerspell on the avatar, and the caller needs to
/// interleave the two.
///
/// What it costs, so it is not a surprise later:
///  * The vocabulary is whatever clips exist; everything else fingerspells.
///  * Clips favour one signer (see authoring/fetch_videos.py) but coverage is
///    not complete, so some sentences change signer partway.
///  * There is no blending between words; one clip ends and the next begins.
struct SignVideoView: UIViewRepresentable {

    var clip: URL?
    var speed: Float

    func makeUIView(context: Context) -> PlayerView {
        let view = PlayerView()
        view.backgroundColor = .black
        view.playerLayer.player = context.coordinator.player
        // Fill: the player is shaped close to the clips' own 3:4 portrait, so
        // filling trims only a sliver top and bottom instead of leaving bars.
        view.playerLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ view: PlayerView, context: Context) {
        context.coordinator.show(clip, speed: speed)
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    static func dismantleUIView(_ view: PlayerView, coordinator: Coordinator) {
        coordinator.player.pause()
        coordinator.player.replaceCurrentItem(with: nil)
    }

    /// A plain UIView backed by AVPlayerLayer. Using the layer directly rather
    /// than AVPlayerViewController keeps the transport controls off screen.
    final class PlayerView: UIView {
        override class var layerClass: AnyClass { AVPlayerLayer.self }
        var playerLayer: AVPlayerLayer { layer as! AVPlayerLayer }
    }

    final class Coordinator {
        let player = AVPlayer()
        private var showing: URL?

        init() {
            // These clips have no audio worth hearing, and taking the audio
            // session would duck whatever the user is already listening to.
            player.isMuted = true
        }

        func show(_ url: URL?, speed: Float) {
            guard url != showing else {
                if player.rate != 0 { player.rate = max(speed, 0.1) }
                return
            }
            showing = url
            guard let url else {
                player.pause()
                player.replaceCurrentItem(with: nil)
                return
            }
            player.replaceCurrentItem(with: AVPlayerItem(url: url))
            player.seek(to: .zero)
            player.rate = max(speed, 0.1)
        }
    }
}

/// Where the bundled clips live, and which glosses have one.
enum SignVideos {

    /// Bundled as a folder reference so 2000 files do not flatten into the app
    /// root and collide with everything else in there.
    private static let root: URL? =
        Bundle.main.url(forResource: "sign_videos", withExtension: nil)

    private static let available: Set<String> = {
        guard let root,
              let names = try? FileManager.default.contentsOfDirectory(atPath: root.path)
        else { return [] }
        return Set(names.filter { $0.hasSuffix(".mp4") }
                        .map { String($0.dropLast(4)).lowercased() })
    }()

    static var count: Int { available.count }

    static func has(_ gloss: String) -> Bool { available.contains(key(gloss)) }

    static func url(for gloss: String) -> URL? {
        guard let root, has(gloss) else { return nil }
        return root.appendingPathComponent(key(gloss) + ".mp4")
    }

    /// How long a clip runs, so playback can wait for it rather than guess.
    /// Falls back to a typical sign length if the asset will not report.
    static func duration(of url: URL) async -> Double {
        let asset = AVURLAsset(url: url)
        guard let d = try? await asset.load(.duration) else { return 1.6 }
        let seconds = CMTimeGetSeconds(d)
        return seconds.isFinite && seconds > 0.05 ? seconds : 1.6
    }

    /// Matches the naming in authoring/fetch_videos.py.
    private static func key(_ gloss: String) -> String {
        gloss.lowercased()
            .replacingOccurrences(of: "/", with: "-")
            .replacingOccurrences(of: " ", with: "_")
    }
}
