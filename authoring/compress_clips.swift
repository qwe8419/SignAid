// Shrink the sign clips for bundling: 360x480, 30 fps, H.264.
//
//     swift authoring/compress_clips.swift sign_videos
//
// PopSign recordings come off phones at up to 1944x2592 and, for some signers,
// 120 fps; even after avconvert's 540p preset the library was heading for
// ~700 MB. avconvert cannot change frame rate, hence this. A sign is legible at
// 480p and 30 fps -- that is what video-relay services run at. Rewrites each
// file in place, and skips files already at or under the target.
import AVFoundation
import CoreImage
import Foundation

let target = CGSize(width: 360, height: 480)
let dir = URL(fileURLWithPath: CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "sign_videos")

func compress(_ src: URL) async throws -> Bool {
    let asset = AVURLAsset(url: src)
    guard let track = try await asset.loadTracks(withMediaType: .video).first else { return false }
    let natural = try await track.load(.naturalSize)
    let transform = try await track.load(.preferredTransform)
    let shown = natural.applying(transform)
    let size = CGSize(width: abs(shown.width), height: abs(shown.height))
    let fps = try await track.load(.nominalFrameRate)
    if size.height <= target.height + 1 && fps <= 31 { return false }

    let scale = min(target.width / size.width, target.height / size.height)
    let out = CGSize(width: (size.width * scale / 2).rounded() * 2,
                     height: (size.height * scale / 2).rounded() * 2)
    let comp = try await AVMutableVideoComposition.videoComposition(with: asset) { req in
        let img = req.sourceImage.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        req.finish(with: img, context: nil)
    }
    comp.renderSize = out
    comp.frameDuration = CMTime(value: 1, timescale: 30)

    let tmp = src.deletingLastPathComponent()
        .appendingPathComponent(".tmp-" + src.lastPathComponent)
    try? FileManager.default.removeItem(at: tmp)
    guard let export = AVAssetExportSession(asset: asset,
                                            presetName: AVAssetExportPresetMediumQuality) else { return false }
    export.videoComposition = comp
    export.shouldOptimizeForNetworkUse = true
    try await export.export(to: tmp, as: .mp4)
    _ = try FileManager.default.replaceItemAt(src, withItemAt: tmp)
    return true
}

let files = (try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil))
    .filter { $0.pathExtension == "mp4" && !$0.lastPathComponent.hasPrefix(".") }
    .sorted { $0.lastPathComponent < $1.lastPathComponent }
var before = 0, after = 0, changed = 0
for f in files {
    let a = (try? f.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
    do {
        if try await compress(f) { changed += 1 }
    } catch {
        print("\(f.lastPathComponent): \(error.localizedDescription)")
    }
    let b = (try? f.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
    before += a; after += b
}
print("\(files.count) clips, \(changed) recompressed: \(before / 1_000_000) MB -> \(after / 1_000_000) MB")
