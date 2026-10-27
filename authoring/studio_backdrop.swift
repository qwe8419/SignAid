// Put every sign clip on the same studio backdrop.
//
//     swift authoring/studio_backdrop.swift <in_dir> <out_dir>
//
// PopSign signers recorded at home, across many sessions: one word is in a
// kitchen, the next in front of a painting. Apple's person segmentation (the
// model behind Portrait mode) cuts the signer out of each frame, and they are
// composited onto one dark backdrop -- the look of an interpreter on a plain
// background, which is also what makes hands easiest to read. Output is
// 360x480 H.264 at the source frame rate, so compress_clips.swift is not
// needed afterwards. Files already in <out_dir> are skipped.
import AVFoundation
import CoreImage
import CoreImage.CIFilterBuiltins
import Foundation
import Vision

let args = CommandLine.arguments
let inDir = URL(fileURLWithPath: args.count > 1 ? args[1] : "sign_videos_raw")
let outDir = URL(fileURLWithPath: args.count > 2 ? args[2] : "sign_videos")
let outSize = CGSize(width: 360, height: 480)
let ctx = CIContext()

/// Charcoal, slightly lighter at the top: reads as a lit backdrop rather than
/// a hole, and matches the app's dark UI.
func backdrop(_ extent: CGRect) -> CIImage {
    let g = CIFilter.linearGradient()
    g.point0 = CGPoint(x: extent.midX, y: extent.maxY)
    g.point1 = CGPoint(x: extent.midX, y: extent.minY)
    g.color0 = CIColor(red: 0.16, green: 0.17, blue: 0.20)
    g.color1 = CIColor(red: 0.06, green: 0.065, blue: 0.08)
    return g.outputImage!.cropped(to: extent)
}

func process(_ src: URL, _ dst: URL) async throws {
    let asset = AVURLAsset(url: src)
    guard let track = try await asset.loadTracks(withMediaType: .video).first else { return }
    let transform = try await track.load(.preferredTransform)
    let natural = try await track.load(.naturalSize)
    let shown = natural.applying(transform)
    let upright = CGSize(width: abs(shown.width), height: abs(shown.height))
    let scale = min(outSize.width / upright.width, outSize.height / upright.height)
    let out = CGSize(width: (upright.width * scale / 2).rounded() * 2,
                     height: (upright.height * scale / 2).rounded() * 2)

    let reader = try AVAssetReader(asset: asset)
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    reader.add(output)

    try? FileManager.default.removeItem(at: dst)
    let writer = try AVAssetWriter(outputURL: dst, fileType: .mp4)
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
        AVVideoCodecKey: AVVideoCodecType.h264,
        AVVideoWidthKey: out.width, AVVideoHeightKey: out.height,
        AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 900_000],
    ])
    input.expectsMediaDataInRealTime = false
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: out.width, kCVPixelBufferHeightKey as String: out.height])
    writer.add(input)
    writer.shouldOptimizeForNetworkUse = true
    reader.startReading()
    writer.startWriting()
    var started = false

    let request = VNGeneratePersonSegmentationRequest()
    request.qualityLevel = .accurate          // fingers are the whole point
    request.outputPixelFormat = kCVPixelFormatType_OneComponent8

    while let sample = output.copyNextSampleBuffer() {
        guard let pb = CMSampleBufferGetImageBuffer(sample) else { continue }
        let pts = CMSampleBufferGetPresentationTimeStamp(sample)
        if !started { writer.startSession(atSourceTime: pts); started = true }

        try VNImageRequestHandler(cvPixelBuffer: pb, options: [:]).perform([request])
        var frame = CIImage(cvPixelBuffer: pb)
        if let maskPB = request.results?.first?.pixelBuffer {
            var mask = CIImage(cvPixelBuffer: maskPB)
            mask = mask.transformed(by: CGAffineTransform(
                scaleX: frame.extent.width / mask.extent.width,
                y: frame.extent.height / mask.extent.height))
            let blend = CIFilter.blendWithMask()
            blend.inputImage = frame
            blend.backgroundImage = backdrop(frame.extent)
            blend.maskImage = mask
            frame = blend.outputImage!.cropped(to: frame.extent)
        }
        // Upright, then down to the output size.
        frame = frame.transformed(by: transform)
        frame = frame.transformed(by: CGAffineTransform(translationX: -frame.extent.minX,
                                                        y: -frame.extent.minY))
        frame = frame.transformed(by: CGAffineTransform(scaleX: scale, y: scale))

        while !input.isReadyForMoreMediaData { try await Task.sleep(nanoseconds: 2_000_000) }
        guard let pool = adaptor.pixelBufferPool else { break }
        var outPB: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &outPB)
        guard let outPB else { break }
        ctx.render(frame, to: outPB)
        adaptor.append(outPB, withPresentationTime: pts)
    }
    input.markAsFinished()
    await writer.finishWriting()
    if writer.status != .completed {
        throw writer.error ?? NSError(domain: "studio", code: 1)
    }
}

try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
let files = try FileManager.default.contentsOfDirectory(at: inDir, includingPropertiesForKeys: nil)
    .filter { $0.pathExtension == "mp4" && !$0.lastPathComponent.hasPrefix(".") }
    .sorted { $0.lastPathComponent < $1.lastPathComponent }
var done = 0
for f in files {
    let dst = outDir.appendingPathComponent(f.lastPathComponent)
    if FileManager.default.fileExists(atPath: dst.path) { continue }
    do {
        try await process(f, dst)
        done += 1
        if done % 20 == 0 { print("\(done) clips") }
    } catch {
        print("\(f.lastPathComponent): \(error.localizedDescription)")
    }
}
print("\(done) clips written to \(outDir.path)")
