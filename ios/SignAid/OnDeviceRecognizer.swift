import AVFoundation
import CoreGraphics
import SwiftUI
import CoreML
import MediaPipeTasksVision
import Vision

/// Letter recognition running entirely on the iPhone, using the EXACT pipeline
/// the Raspberry Pi runs: MediaPipe hand landmarks (x, y and z) -> the same 298
/// features -> the same 3-seed ensemble (0.952 val, 0.927 stress).
///
/// This previously used Apple's Vision framework instead. Vision returns 2D
/// landmarks with a bottom-left origin and its own handedness conventions,
/// which meant a retrained 2D model (0.928), a hand-mapped joint order, a
/// y-flip, and an orientation flag whose meaning is ambiguous. Every
/// recognition bug on the phone came from one of those mismatches -- including
/// H being read as R, which is a chirality error. Running the same tracker as
/// training removes the whole class of problem rather than patching each one.
@MainActor
final class OnDeviceRecognizer: NSObject, ObservableObject {

    @Published var text = ""
    @Published var liveLabel = ""
    @Published var liveConfidence: Double = 0
    @Published var progress: Double = 0
    @Published var handVisible = false
    @Published var running = false
    @Published var justCommitted: String?
    @Published var commitCount = 0
    @Published var fps: Double = 0
    /// Surfaced so the UI can say "seen but not confident" instead of just
    /// sitting there — the failure mode that looked like a hang.
    @Published var belowThreshold = false
    /// Non-nil when setup or per-frame conversion failed. Without this a broken
    /// pipeline just looks like "no hand", which is indistinguishable from
    /// working-but-empty.
    @Published var trackerError: String?
    /// Top alternatives for the current frame, so a correct-but-low-confidence
    /// read is visible and selectable instead of invisible.
    @Published var alternatives: [Alternative] = []

    struct Alternative: Identifiable {
        let id = UUID()
        let letter: String
        let confidence: Double
    }
    /// Latest raw landmarks. Deliberately NOT @Published: it changes 30 times a
    /// second, and publishing it re-rendered every observing view at that rate,
    /// which froze the collect screen (a wheel Picker and a 26-cell grid cannot
    /// be laid out 30x/sec).
    private(set) var lastLandmarks: [[Double]]?
    /// The same hand in the model's frame, for Practice's hand-shape coach.
    /// Not published, for the same reason as lastLandmarks.
    private(set) var lastModelPoints: [[Double]]?

    // --- data collection ---
    // Owned here, not in the view. A SwiftUI View is a struct, so a long-running
    // Task that mutates its @State works on a copy: the loop never observes the
    // stop flag and spins forever, which froze the collect screen.
    @Published private(set) var collecting = false
    @Published private(set) var collectedCount = 0
    private var collectBuffer: [[[Double]]] = []
    private var collectTarget = 25
    private var lastCollectAt: CFTimeInterval = 0
    /// ~12 samples a second. Adjacent frames at 30fps are near-identical and add
    /// volume without adding information.
    private let collectInterval: CFTimeInterval = 0.08

    func startCollecting(target: Int) {
        collectTarget = target
        collectBuffer.removeAll()
        collectedCount = 0
        lastCollectAt = 0
        collecting = true
    }

    /// Where recorded bursts land: the app's own Documents directory.
    ///
    /// Collection used to only upload to the Mac server, so it did nothing
    /// unless that server happened to be running — which is the opposite of the
    /// point, since the whole reason to collect on the phone is to capture the
    /// phone's own camera, distance and grip. Saving locally means recording
    /// works anywhere; the files are pulled off afterwards with
    ///   xcrun devicectl device copy from --domain-type appDataContainer \
    ///        --domain-identifier com.vihaan.signaid --source Documents/phone_letters
    static var collectionDirectory: URL {
        let docs = FileManager.default.urls(for: .documentDirectory,
                                            in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("phone_letters", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir,
                                                 withIntermediateDirectories: true)
        return dir
    }

    /// Write one burst next to the others. Landmarks are stored in the same
    /// convention the training set uses, so the two merge without further
    /// correction.
    @discardableResult
    func saveBurst(letter: String, burst: Int, variation: String,
                   samples: [[[Double]]]) -> Bool {
        let payload: [String: Any] = [
            "letter": letter, "burst": burst, "variation": variation,
            "source": "phone", "captured": ISO8601DateFormatter().string(from: Date()),
            // Samples are already in the model's frame (HandFeatures.toModelFrame),
            // so train_robust.py loads them as-is. Older files lack this key.
            "frame": "model",
            "samples": samples,
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: payload) else {
            return false
        }
        let stamp = Int(Date().timeIntervalSince1970 * 1000)
        let url = Self.collectionDirectory
            .appendingPathComponent("\(letter)_\(burst)_\(stamp).json")
        return (try? data.write(to: url)) != nil
    }

    /// How many samples are stored per letter, read back off disk so the count
    /// survives the app being closed.
    static func savedCounts() -> [String: Int] {
        var out: [String: Int] = [:]
        let files = (try? FileManager.default.contentsOfDirectory(
            at: collectionDirectory, includingPropertiesForKeys: nil)) ?? []
        for f in files where f.pathExtension == "json" {
            guard let d = try? Data(contentsOf: f),
                  let o = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
                  let letter = o["letter"] as? String,
                  let s = o["samples"] as? [[[Double]]] else { continue }
            out[letter, default: 0] += s.count
        }
        return out
    }

    /// Returns whatever was captured and resets. Safe to call at any time.
    @discardableResult
    func stopCollecting() -> [[[Double]]] {
        collecting = false
        let out = collectBuffer
        collectBuffer.removeAll()
        collectedCount = 0
        return out
    }
    /// The exact image handed to the crop classifier. Shown in the UI so a
    /// misaligned crop is visible instead of silently poisoning the fusion.
    @Published var debugCrop: UIImage?
    @Published var cropLabel = ""
    /// Whether the last frame actually fused, so a silently-rejected crop is
    /// visible rather than looking like the image model is simply not helping.
    @Published var fusionState = "off"
    /// Set when the hand is too small in frame for the fine distinctions to
    /// survive. Cheaper to tell the user to move closer than to guess.
    @Published var handTooSmall = false
    /// On by default: the crop is the thing most likely to be subtly wrong,
    /// and a thumbnail showing the hand is the only cheap way to confirm it.
    nonisolated(unsafe) var wantsDebugCrop = true
    /// Whether to mirror landmarks before recognition. Exposed because the
    /// correct value depends on camera and on how the training data was
    /// captured, and it is far quicker to settle by trying it than to reason
    /// about three layers of coordinate conventions.
    @Published var mirrorLandmarks = true {
        didSet { mirrorFlag = mirrorLandmarks; candidate = nil; count = 0; progress = 0; voteHistory.removeAll() }
    }
    private nonisolated(unsafe) var mirrorFlag = true

    var autoCommit = true
    /// Calibrated against FUSED confidence on the held-out bursts: mean 0.939,
    /// and 0.45 keeps 98.8% of predictions at 0.991 accuracy. The old 0.55 was
    /// inherited from the landmark-only model and blocked correct reads
    /// (Z 8/33, E 3/50, Q 3/44).
    /// 0.35, re-derived for the phone rather than inherited from the glasses.
    ///
    /// The glasses use 0.50, and that is right for them: on webcam captures the
    /// model's mean confidence is 0.942 and 99% of frames clear the gate. On
    /// 800 letters captured through the phone it is 0.391, and only 24% clear
    /// it — so the same threshold blocked three quarters of frames, including
    /// correct ones, and 22 of 32 recorded bursts never committed at all even
    /// where the per-frame prediction was right every time.
    ///
    /// Swept against those captures (test_shipped.py):
    ///     0.50 -> commits 25%, right 0.62 when it does
    ///     0.35 -> commits 59%, right 0.84
    ///     0.25 -> commits 81%, right 0.69
    /// 0.25 scores highest overall but is wrong on nearly a third of commits.
    /// A wrong letter costs a delete and a retry; a non-commit costs holding
    /// still a moment longer, and the top-3 row is tappable. Precision wins.
    ///
    /// Back to 0.50 now the phone feeds the model in its training frame
    /// (HandFeatures.toModelFrame): mean confidence on the phone captures rose
    /// from 0.39 to 0.92, so the strict gate no longer blocks correct reads and
    /// buys precision.
    var acceptMinConfidence = 0.50
    /// Share of the fused score from the image model. Swept on the held-out
    /// bursts against the 2D landmark net: 0.0 -> 0.928, 0.5 -> 0.973,
    /// 0.7 -> 0.983, 1.0 -> 0.981. The image model resolves exactly what
    /// landmarks cannot -- crossed and touching fingers -- taking R from 0.500
    /// to 1.000, U 0.700 -> 1.000, D 0.640 -> 1.000, X 0.680 -> 0.980.
    /// 0 = landmark only (measured 0.928 and known to behave on-device).
    /// OFF, matching the glasses.
    ///
    /// The crop model scores 0.995 on held-out bursts and fixes exactly the
    /// letters landmarks lose, which is why it looked worth switching on. But
    /// eval_lighting.py had already answered this: landmarks hold 0.949 in EVERY
    /// lighting condition while the crop model swings between 0.097 and 0.981.
    /// It is excellent in the light it was captured in and worthless elsewhere,
    /// and pi_standalone.py dropped it for that reason. On the phone it read B
    /// as H. Available behind the toggle; not the default.
    @Published var useImageModel = false {
        didSet { wantsFusion = useImageModel }
    }
    /// Readable from the capture queue, which cannot touch @Published state.
    private nonisolated(unsafe) var wantsFusion = false
    var fusionWeight: Float { useImageModel ? 0.7 : 0.0 }
    @Published var usingFrontCamera = true
    /// Mirror of `usingFrontCamera` readable from the capture queue.
    private nonisolated(unsafe) var frontFacing = true
    /// Frames a letter must hold before it is typed, from Settings ▸ Letter
    /// speed. 12 frames is about 0.4 s at 30 fps: long enough that the shapes a
    /// hand passes through on the way between letters do not get typed.
    private var voteWindow: Int {
        switch UserDefaults.standard.string(forKey: "letterSpeed") ?? "normal" {
        case "quick": return 8
        case "relaxed": return 16
        default: return 12
        }
    }
    private let minAgreement = 0.6

    // Re-arming: when the SAME letter may be typed again.
    //
    // It used to re-arm whenever a single frame disagreed with the last letter,
    // or the tracker lost the hand for 4 frames -- both happen constantly while
    // holding a sign still, so held letters typed themselves again and again.
    // Now a repeat needs a deliberate act, the same ones ASL uses:
    //  * take the hand away for `releaseSeconds`, or
    //  * slide/bounce the hand (how signers show a double letter, as in HELLO):
    //    the wrist moves `repeatMove` hand-lengths and then settles again.
    // A DIFFERENT letter never needs re-arming; it only has to win a full window.
    private let releaseSeconds: CFTimeInterval = 0.5
    private let repeatMove = 0.45
    /// Hand must be this still (wrist spread / hand length) over `steadyFrames`
    /// to type anything, so a letter is read when it is held, not in passing.
    private let steadyTolerance = 0.12
    private let steadyFrames = 6
    private let minCommitGap: CFTimeInterval = 0.35
    /// Letters defined by movement are exempt from the stillness rule.
    private let motionLetters: Set<String> = ["J", "Z"]
    private var handAwaySince: CFTimeInterval?
    private var wristTrail: [(Double, Double)] = []
    private var handLength = 0.1
    private var commitWrist: (Double, Double)?
    private var movedSinceCommit = false
    private var lastCommitTime: CFTimeInterval = 0
    /// Recent (label, confidence) pairs, the window the vote runs over.
    private var voteHistory: [(String, Double)] = []

    let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "signaid.camera", qos: .userInitiated)
    private let output = AVCaptureVideoDataOutput()
    private var net: LocalLetterNet?
    private nonisolated(unsafe) var landmarker: HandLandmarker?
    /// Read from the capture queue, so it is nonisolated. Assigned once in
    /// init and never mutated afterwards.
    private nonisolated(unsafe) var cropModel: VNCoreMLModel?
    private nonisolated let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    private nonisolated(unsafe) var nextStamp = 0

    /// Margin around the hand, as a fraction of its longer side.
    ///
    /// collect_letters.py padded the landmark bounding box by a flat 40 px on
    /// 640x480 frames. Measured across 200 of those crops, that works out to
    /// 0.34 of the hand's longer side. Expressed as a fraction it transfers to
    /// the phone's much larger frames, where a flat 40 px would be a far
    /// tighter crop than the model has ever seen.
    private nonisolated let cropPadFraction: CGFloat = 0.34

    /// Crop the frame to the hand, the way the training data was cut.
    ///
    /// Landmarks arrive normalised against the ORIENTED image, so the buffer is
    /// oriented first and the box computed in that space. CoreImage's origin is
    /// bottom-left where MediaPipe's is top-left, which is what the y flip is.
    private nonisolated func handCrop(_ buffer: CMSampleBuffer,
                                      landmarks: [[Double]],
                                      mirrored: Bool) -> CGImage? {
        guard let pixels = CMSampleBufferGetImageBuffer(buffer) else { return nil }
        var image = CIImage(cvPixelBuffer: pixels).oriented(.right)
        let extent = image.extent
        guard extent.width > 1, extent.height > 1 else { return nil }

        let xs = landmarks.map { CGFloat($0[0]) }, ys = landmarks.map { CGFloat($0[1]) }
        guard let minX = xs.min(), let maxX = xs.max(),
              let minY = ys.min(), let maxY = ys.max() else { return nil }

        var x0 = minX * extent.width, x1 = maxX * extent.width
        // Top-left normalised y -> bottom-left CoreImage y.
        var y0 = (1 - maxY) * extent.height, y1 = (1 - minY) * extent.height
        let pad = max(x1 - x0, y1 - y0) * cropPadFraction
        x0 -= pad; x1 += pad; y0 -= pad; y1 += pad

        let box = CGRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0)
            .intersection(extent)
        guard box.width > 8, box.height > 8 else { return nil }

        image = image.cropped(to: box)
        if mirrored {
            // collect_letters.py ran cv2.flip(frame, 1) before cropping, so the
            // model has only ever seen mirrored hands.
            image = image.transformed(by: CGAffineTransform(scaleX: -1, y: 1))
        }
        return ciContext.createCGImage(image, from: image.extent)
    }

    /// Class probabilities from the crop model, in the same A-Z order as the
    /// landmark model. Returns nil rather than guessing if anything is off.
    private nonisolated func cropProbabilities(_ image: CGImage) -> [Float]? {
        guard let cropModel else { return nil }
        let request = VNCoreMLRequest(model: cropModel)
        // Scale-fill, not aspect-fit: the training crops were resized straight
        // to a square without preserving aspect, so letterboxing here would show
        // the model a shape it has never seen.
        request.imageCropAndScaleOption = .scaleFill
        guard (try? VNImageRequestHandler(cgImage: image, options: [:])
                .perform([request])) != nil else { return nil }

        // The network exposes both logits and softmax. Take the probabilities:
        // fusing raw logits with probabilities would let one term dominate
        // entirely, since the logits run to about +/- 14.
        for observation in request.results ?? [] {
            if let features = observation as? VNCoreMLFeatureValueObservation,
               let array = features.featureValue.multiArrayValue,
               array.count == 26 {
                var out = [Float](repeating: 0, count: 26)
                var sum: Float = 0
                for i in 0..<26 {
                    out[i] = array[i].floatValue
                    sum += out[i]
                }
                // Only the softmax output sums to one; skip the logits head.
                if abs(sum - 1) < 0.05 { return out }
            }
        }
        return nil
    }

    private var candidate: String?
    private var count = 0
    private var disagreements = 0
    private var armed = true
    private var lastCommitted: String?
    private var frameTimes: [CFTimeInterval] = []
    private var lastFPSPublish: CFTimeInterval = 0
    private var recent: [[Float]] = []
    private let smoothingWindow = 5
    private let tap = UIImpactFeedbackGenerator(style: .medium)

    override init() {
        super.init()
        net = LocalLetterNet()
        if let path = Bundle.main.path(forResource: "hand_landmarker", ofType: "task") {
            let options = HandLandmarkerOptions()
            options.baseOptions.modelAssetPath = path
            options.numHands = 1
            options.runningMode = .video
            // MediaPipe's defaults, which is what the glasses run —
            // pi_standalone.py sets none of these. Lowering them to 0.4 let
            // marginal detections through, and a weak detection produces weak
            // landmarks rather than no landmarks, so the letter model was being
            // asked to classify hands MediaPipe was not confident it had found.
            options.minHandDetectionConfidence = 0.5
            options.minTrackingConfidence = 0.5
            options.minHandPresenceConfidence = 0.5
            do {
                landmarker = try HandLandmarker(options: options)
            } catch {
                trackerError = "Hand tracker failed: \(error.localizedDescription)"
            }
        } else {
            trackerError = "hand_landmarker.task missing from the app bundle"
        }
        if net == nil { trackerError = "letter model failed to load" }
        if let url = Bundle.main.url(forResource: "LetterCrop", withExtension: "mlmodelc"),
           let ml = try? MLModel(contentsOf: url) {
            cropModel = try? VNCoreMLModel(for: ml)
        }
    }

    // MARK: - Camera

    func start() {
        guard !running else { return }
        AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
            guard granted else { return }
            Task { @MainActor in self?.configureAndRun() }
        }
    }

    private func configureAndRun() {
        if session.inputs.isEmpty {
            session.beginConfiguration()
            // VGA, and it must stay 4:3.
            //
            // The hand IS too small at arm's length — held there it spans a
            // quarter to a half of what it did on the training webcam, and
            // MediaPipe stops being able to tell touching fingers from
            // separated ones, which is the whole difference between U, H and R.
            //
            // But raising this to .hd1920x1080 to buy detail made everything
            // read as H, because 1080p is 16:9 where VGA is 4:3. MediaPipe
            // normalises landmarks against the frame it is handed, so changing
            // the aspect ratio changes every coordinate the model sees. The
            // extra pixels have to come from ZOOM, which crops the sensor and
            // leaves the aspect alone.
            session.sessionPreset = .vga640x480
            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video,
                                                       position: usingFrontCamera ? .front : .back),
                  let input = try? AVCaptureDeviceInput(device: device),
                  session.canAddInput(input) else {
                session.commitConfiguration(); return
            }
            session.addInput(input)
            output.setSampleBufferDelegate(self, queue: queue)
            output.alwaysDiscardsLateVideoFrames = true    // never queue stale frames
            // MPImage(sampleBuffer:) only accepts 32BGRA; the capture default is
            // bi-planar YUV, which makes the initialiser return nil every frame.
            output.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]
            if session.canAddOutput(output) { session.addOutput(output) }
            session.commitConfiguration()
        }
        running = true
        tap.prepare()
        queue.async { [session] in if !session.isRunning { session.startRunning() } }
    }

    /// Front camera for signing at yourself, back camera for signing at someone
    /// else -- the crop model does not care, but the mirroring does.
    func flipCamera() {
        usingFrontCamera.toggle()
        frontFacing = usingFrontCamera
        session.beginConfiguration()
        session.inputs.forEach { session.removeInput($0) }
        if let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video,
                                                position: usingFrontCamera ? .front : .back),
           let input = try? AVCaptureDeviceInput(device: device),
           session.canAddInput(input) {
            session.addInput(input)
        }
        session.commitConfiguration()
        candidate = nil; count = 0; progress = 0; voteHistory.removeAll()
    }

    func stop() {
        running = false
        queue.async { [session] in if session.isRunning { session.stopRunning() } }
        liveLabel = ""; progress = 0; handVisible = false
    }

    // MARK: - Glasses as the camera

    /// True while frames are arriving from the glasses.
    @Published var glassesConnected = false
    /// The glasses' frames for the preview. Separate object on purpose: frames
    /// arrive 30 times a second, and publishing them from the recogniser would
    /// re-render every view that observes it.
    let glassesFeed = GlassesFeed()
    private let glassesStream = GlassesStream()
    private nonisolated let glassesQueue = DispatchQueue(label: "signaid.glasses",
                                                         qos: .userInitiated)
    private nonisolated(unsafe) var glassesBusy = false
    private var glassesActive = false
    /// The glasses' webcam delivers upright landscape frames, the same frame the
    /// model was trained in, so their points skip the phone's sensor rotation.
    private var pointsInModelFrame = false

    /// Use the glasses as the camera. The Pi only streams video (a Pi 4 doing
    /// hand tracking overheated and stuttered); this phone runs MediaPipe, the
    /// letter model and the typing rules on each frame -- the same code as its
    /// own camera -- so the glasses behave exactly like the phone.
    func startGlasses(cameraURL url: URL) {
        stop()
        stopGlasses()
        pointsInModelFrame = true
        glassesActive = true
        resetRecognition()
        glassesStream.onJPEG = { [weak self] jpeg in
            guard let self, !self.glassesBusy else { return }   // newest only
            self.glassesBusy = true
            self.glassesQueue.async { self.processGlassesFrame(jpeg) }
        }
        glassesStream.onEnd = { [weak self] in
            Task { @MainActor in
                guard let self, self.glassesActive else { return }
                self.glassesConnected = false
                self.handle(nil, cropProbs: nil)
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                if self.glassesActive { self.glassesStream.start(url) }   // reconnect
            }
        }
        glassesStream.start(url)
    }

    private nonisolated func processGlassesFrame(_ jpeg: Data) {
        defer { glassesBusy = false }
        guard let image = UIImage(data: jpeg) else { return }
        var points: [[Double]]?
        if let landmarker, let mpImage = try? MPImage(uiImage: image) {
            let stamp = max(Int(CACurrentMediaTime() * 1000), nextStamp)
            nextStamp = stamp + 1
            if let result = try? landmarker.detect(videoFrame: mpImage,
                                                   timestampInMilliseconds: stamp),
               let hand = result.landmarks.first, hand.count == 21 {
                points = hand.map { [Double($0.x), Double($0.y), Double($0.z)] }
            }
        }
        let pts = points
        Task { @MainActor [weak self] in
            guard let self, self.glassesActive else { return }
            self.glassesFeed.image = image
            self.glassesConnected = true
            self.handle(pts, cropProbs: nil)
        }
    }

    func stopGlasses() {
        glassesActive = false
        glassesStream.stop()
        glassesFeed.image = nil
        glassesConnected = false
        pointsInModelFrame = false
        handVisible = false
        liveLabel = ""; progress = 0
    }

    private func resetRecognition() {
        voteHistory.removeAll(); recent.removeAll()
        progress = 0; candidate = nil; count = 0
    }

    // MARK: - Text editing

    /// Append a letter chosen from the alternatives list. Lets a correct read
    /// that the gate rejected still be used, without lowering the gate.
    func append(_ letter: String) {
        text += letter
        lastCommitted = letter
        armed = false
        markCommitted()
        candidate = nil
        count = 0
        progress = 0
        tap.impactOccurred()
        commitCount += 1
        justCommitted = letter
        let token = commitCount
        Task {
            try? await Task.sleep(nanoseconds: 900_000_000)
            if self.commitCount == token { self.justCommitted = nil }
        }
    }

    func clear() { text = "" }
    func delete() { if !text.isEmpty { text.removeLast() } }
    func space() { if !text.isEmpty && !text.hasSuffix(" ") { text += " " } }

    // MARK: - Recognition

    fileprivate func handle(_ points: [[Double]]?, cropProbs: [Float]?) {
        let now = CACurrentMediaTime()
        frameTimes.append(now)
        frameTimes.removeAll { now - $0 > 2 }
        if now - lastFPSPublish > 0.5 {          // twice a second is plenty
            lastFPSPublish = now
            fps = Double(frameTimes.count) / 2.0
        }

        guard let points, let net else {
            handVisible = false
            liveLabel = ""; liveConfidence = 0; candidate = nil; count = 0
            lastLandmarks = nil
            alternatives = []
            recent.removeAll()
            if handAwaySince == nil { handAwaySince = now }
            // A dropped frame or two is the tracker blinking, not the hand
            // leaving; only a real absence clears the stillness history.
            if let since = handAwaySince, now - since >= releaseSeconds {
                armed = true
                wristTrail.removeAll()
            }
            progress = 0
            return
        }

        handVisible = true
        handAwaySince = nil
        trackMotion(points)

        // Wrist to middle knuckle, in frame units. The training captures sit
        // around 0.5; below about 0.25 the two-finger letters start collapsing
        // into each other because there is not enough resolution on the hand.
        if points.count >= 10 {
            let dx = points[9][0] - points[0][0], dy = points[9][1] - points[0][1]
            handTooSmall = (dx * dx + dy * dy).squareRoot() < 0.25
        }

        lastLandmarks = points

        if collecting {
            let now = CACurrentMediaTime()
            if now - lastCollectAt >= collectInterval {
                lastCollectAt = now
                collectBuffer.append(pointsInModelFrame ? points : HandFeatures.toModelFrame(points))
                if collectBuffer.count != collectedCount {
                    collectedCount = collectBuffer.count
                }
                if collectBuffer.count >= collectTarget { collecting = false }
            }
        }
        if collecting {
            // No need to classify while recording training data, and skipping it
            // avoids publishing liveLabel/confidence 30x a second.
            return
        }
        let modelPoints = pointsInModelFrame ? points : HandFeatures.toModelFrame(points)
        lastModelPoints = modelPoints
        var probs = net.probs(HandFeatures.extract(modelPoints))

        // Average the last few frames' probabilities. The Pi gets this
        // stabilising effect from its 9-frame vote; without it a single noisy
        // frame can swing the label and reset the commit counter.
        recent.append(probs)
        if recent.count > smoothingWindow { recent.removeFirst() }
        if recent.count > 1 {
            var avg = [Float](repeating: 0, count: probs.count)
            for frame in recent {
                for i in avg.indices { avg[i] += frame[i] }
            }
            for i in avg.indices { avg[i] /= Float(recent.count) }
            probs = avg
        }
        // Fuse with the image model when it produced something. Landmarks are
        // geometry only and cannot see fingers touching or crossing; the crop
        // classifier can.
        // Fused only when the two models are in the same neighbourhood. The
        // image model is worth 0.7 of the answer when its crop is good and
        // actively destructive when it is not — a misplaced crop classifies
        // background with high confidence and drags the result with it. A crop
        // showing the hand will land inside the landmark model's top three even
        // when the two disagree on which letter it is; one showing a wall will
        // not. So a bad crop degrades to landmark-only rather than to noise.
        if useImageModel, let crop = cropProbs, crop.count == probs.count,
           let cropTop = crop.indices.max(by: { crop[$0] < crop[$1] }) {
            let landmarkTop3 = probs.indices
                .sorted { probs[$0] > probs[$1] }.prefix(3)
            if landmarkTop3.contains(cropTop) {
                let w = fusionWeight
                for i in probs.indices { probs[i] = w * crop[i] + (1 - w) * probs[i] }
                fusionState = "fused"
            } else {
                fusionState = "crop rejected"
            }
        } else {
            fusionState = useImageModel ? "no crop" : "off"
        }
        if let crop = cropProbs, crop.count == net.classes.count,
           let cb = crop.indices.max(by: { crop[$0] < crop[$1] }) {
            cropLabel = "\(net.classes[cb]) \(Int(crop[cb] * 100))%"
        } else {
            cropLabel = "—"
        }
        guard let best = probs.indices.max(by: { probs[$0] < probs[$1] }) else { return }
        let label = net.classes[best]
        let conf = Double(probs[best])
        liveLabel = label
        liveConfidence = conf

        // Top 3, so a correct letter sitting at 28% is visible rather than lost.
        alternatives = probs.indices.sorted { probs[$0] > probs[$1] }.prefix(3)
            .map { Alternative(letter: net.classes[$0], confidence: Double(probs[$0])) }


        guard autoCommit else {
            belowThreshold = false
            voteHistory.removeAll(); progress = 0
            return
        }

        // The glasses' voting scheme, ported wholesale (pi_standalone.py vote()).
        //
        // The phone used to require `stableFrames` CONSECUTIVE agreeing frames
        // and reset the run on disagreement. At 20-30 fps that is brittle: a
        // couple of bad frames mid-letter throws away all the accumulated
        // evidence, so a letter can sit in the ring never committing, and a
        // single confident wrong frame can start a fresh run of its own.
        //
        // Voting over a WINDOW is steadier. Every frame contributes its
        // confidence to its own label, the label with the most accumulated
        // confidence wins, and it only commits if it also carries a majority of
        // the frames. Wrong frames are outvoted rather than being able to reset
        // anything.
        voteHistory.append((label, conf))
        if voteHistory.count > voteWindow {
            voteHistory.removeFirst(voteHistory.count - voteWindow)
        }

        var weight: [String: Double] = [:]
        var frames: [String: Int] = [:]
        for (l, c) in voteHistory {
            weight[l, default: 0] += c
            frames[l, default: 0] += 1
        }
        guard let winner = weight.max(by: { $0.value < $1.value })?.key,
              let hits = frames[winner], !voteHistory.isEmpty else { return }
        let meanConf = weight[winner]! / Double(hits)
        let agreement = Double(hits) / Double(voteHistory.count)

        belowThreshold = meanConf < acceptMinConfidence
        let isRepeat = winner == lastCommitted
        // A held letter shows an empty ring rather than refilling, so it is
        // clear it will not type again until the hand slides or drops.
        progress = isRepeat && !armed ? 0 : min(Double(hits) / Double(voteWindow), 1)

        let qualifies = voteHistory.count >= voteWindow
            && meanConf >= acceptMinConfidence && agreement >= minAgreement
        let steadyEnough = isSteady || motionLetters.contains(winner)
        guard qualifies, steadyEnough, !isRepeat || armed,
              CACurrentMediaTime() - lastCommitTime >= minCommitGap else { return }

        text += winner
        lastCommitted = winner
        armed = false
        markCommitted()
        voteHistory.removeAll()
        progress = 0
        candidate = nil
        count = 0
    }
}

extension OnDeviceRecognizer {
    /// Wrist position and hand length, for the stillness and slide-to-repeat
    /// rules. In frame units; only ratios to the hand's own length are used, so
    /// distance from the camera does not matter.
    fileprivate func trackMotion(_ points: [[Double]]) {
        guard points.count >= 10 else { return }
        let wrist = (points[0][0], points[0][1])
        let len = hypot(points[9][0] - points[0][0], points[9][1] - points[0][1])
        if len > 1e-3 { handLength = len }
        wristTrail.append(wrist)
        if wristTrail.count > steadyFrames { wristTrail.removeFirst(wristTrail.count - steadyFrames) }

        // Slide-to-repeat: moved far enough from where the last letter was
        // typed, then held still again.
        if !armed, let anchor = commitWrist {
            let moved = hypot(wrist.0 - anchor.0, wrist.1 - anchor.1) / handLength
            if moved >= repeatMove { movedSinceCommit = true }
            if movedSinceCommit && isSteady {
                armed = true
                voteHistory.removeAll()       // the repeat must earn a full window
            }
        }
    }

    /// True when the wrist has stayed within `steadyTolerance` hand-lengths of
    /// its average position over the last `steadyFrames` frames.
    fileprivate var isSteady: Bool {
        guard wristTrail.count >= steadyFrames else { return false }
        let mx = wristTrail.map(\.0).reduce(0, +) / Double(wristTrail.count)
        let my = wristTrail.map(\.1).reduce(0, +) / Double(wristTrail.count)
        let spread = wristTrail.map { hypot($0.0 - mx, $0.1 - my) }.max() ?? 0
        return spread / handLength <= steadyTolerance
    }

    fileprivate func markCommitted() {
        lastCommitTime = CACurrentMediaTime()
        commitWrist = wristTrail.last
        movedSinceCommit = false
    }
}

extension OnDeviceRecognizer: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(_ out: AVCaptureOutput,
                                   didOutput buffer: CMSampleBuffer,
                                   from connection: AVCaptureConnection) {
        guard let landmarker else { return }
        // Deliberately never a mirrored orientation. extract() canonicalises
        // rotation (it rotates wrist->middle-MCP to point up), so in-plane
        // rotation cannot affect the features at all -- the ONLY thing this flag
        // changes is chirality. Using .leftMirrored here mirrored the hand, and
        // then mirrorFlag mirrored it again, cancelling out and feeding the model
        // the wrong handedness. Chirality is now controlled in exactly one place.
        let orientation: UIImage.Orientation = .right
        guard let image = try? MPImage(sampleBuffer: buffer, orientation: orientation) else {
            Task { @MainActor [weak self] in self?.trackerError = "Camera format not supported" }
            return
        }

        // MediaPipe's VIDEO mode wants a monotonically increasing timestamp; it
        // tracks between frames instead of re-detecting the palm every time.
        let stamp = Int(CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(buffer)) * 1000)
        let result: HandLandmarkerResult
        do {
            result = try landmarker.detect(videoFrame: image,
                                           timestampInMilliseconds: max(stamp, nextStamp))
        } catch {
            let message = error.localizedDescription
            Task { @MainActor [weak self] in
                self?.trackerError = "detect failed: \(message)"
                self?.handle(nil, cropProbs: nil)
            }
            return
        }
        nextStamp = max(stamp, nextStamp) + 1

        // NO aspect correction. Reverted, twice now.
        //
        // The theory is that MediaPipe normalises against the portrait frame it
        // is handed, so x and y are divided by swapped denominators versus the
        // 640x480 landscape training capture. Re-framing captured frames into
        // portrait and re-running MediaPipe supports it: 0.63 raw vs 0.75 with
        // an x scale of 0.5625, across 16 letters.
        //
        // On the actual phone it does the opposite — nearly everything reads as
        // H. So the simulation is not faithful to the device on this point, most
        // likely because MPImage(sampleBuffer:orientation:) normalises against
        // the ORIGINAL buffer rather than the rotated one, which would mean the
        // landmarks already arrive in the landscape convention and there is
        // nothing to correct.
        //
        // Not re-derived from theory again. The next attempt needs landmarks
        // recorded by a build with NO correction applied, so a sweep over the
        // scale factor measures the device instead of measuring an earlier
        // correction against itself.
        var points: [[Double]]?
        // Kept UNMIRRORED. The landmarks handed to the letter model are
        // mirrored, but the image buffer is not, so locating the crop with
        // mirrored coordinates puts the box on the opposite side of the frame —
        // which cropped background instead of the hand and had the image model
        // confidently classifying noise.
        var framePoints: [[Double]]?
        if let hand = result.landmarks.first, hand.count == 21 {
            // Identical layout to the Python pipeline: normalised x, y and a
            // wrist-relative z, in MediaPipe's own joint order.
            var p = hand.map { [Double($0.x), Double($0.y), Double($0.z)] }
            framePoints = p
            if mirrorFlag {
                // collect_letters.py ran cv2.flip(frame, 1) before extracting
                // landmarks, so the model expects a mirrored hand.
                p = p.map { [1.0 - $0[0], $0[1], $0[2]] }
            }
            points = p
        }
        // The image model only runs when there is a hand to crop to, which is
        // also the only time it could contribute anything.
        var cropProbs: [Float]?
        var thumbnail: CGImage?
        if let fp = framePoints, wantsFusion || wantsDebugCrop,
           let cg = handCrop(buffer, landmarks: fp, mirrored: mirrorFlag) {
            cropProbs = wantsFusion ? cropProbabilities(cg) : nil
            if wantsDebugCrop { thumbnail = cg }
        }
        let probs = cropProbs
        let thumb = thumbnail
        Task { @MainActor [weak self] in
            if let thumb { self?.debugCrop = UIImage(cgImage: thumb) }
            self?.handle(points, cropProbs: probs)
        }
    }
}

/// Loads the exported weights and runs the ensemble: three small MLPs, each
/// three matrix multiplies with BatchNorm already folded into the weights.
final class LocalLetterNet {
    let classes: [String]
    private let mean: [Float], std: [Float]
    private var layers: [(w0: [Float], b0: [Float], w1: [Float], b1: [Float],
                          w2: [Float], b2: [Float], h0: Int, h1: Int, nIn: Int, nOut: Int)] = []

    init?() {
        guard let metaURL = Bundle.main.url(forResource: "letter_net_3d_meta",
                                            withExtension: "json"),
              let binURL = Bundle.main.url(forResource: "letter_net_3d",
                                           withExtension: "bin"),
              let metaData = try? Data(contentsOf: metaURL),
              let bin = try? Data(contentsOf: binURL),
              let meta = try? JSONSerialization.jsonObject(with: metaData) as? [String: Any],
              let classes = meta["classes"] as? [String],
              let tensors = meta["tensors"] as? [[String: Any]],
              let nModels = meta["n_models"] as? Int
        else { return nil }

        self.classes = classes
        let floats: [Float] = bin.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self))
        }
        var cursor = 0
        var store: [String: (values: [Float], shape: [Int])] = [:]
        for t in tensors {
            guard let name = t["name"] as? String, let shape = t["shape"] as? [Int] else { continue }
            let n = shape.reduce(1, *)
            store[name] = (Array(floats[cursor..<(cursor + n)]), shape)
            cursor += n
        }
        guard let m = store["mean"], let s = store["std"] else { return nil }
        mean = m.values
        std = s.values
        for i in 0..<nModels {
            guard let w0 = store["w0_\(i)"], let b0 = store["b0_\(i)"],
                  let w1 = store["w1_\(i)"], let b1 = store["b1_\(i)"],
                  let w2 = store["w2_\(i)"], let b2 = store["b2_\(i)"] else { return nil }
            layers.append((w0.values, b0.values, w1.values, b1.values, w2.values, b2.values,
                           w0.shape[0], w1.shape[0], w0.shape[1], w2.shape[0]))
        }
    }

    /// row-major (out, in) matrix times vector
    private func matvec(_ w: [Float], _ x: [Float], _ b: [Float],
                        _ outDim: Int, _ inDim: Int, relu: Bool) -> [Float] {
        var y = [Float](repeating: 0, count: outDim)
        w.withUnsafeBufferPointer { wp in
            x.withUnsafeBufferPointer { xp in
                for o in 0..<outDim {
                    var acc: Float = b[o]
                    let base = o * inDim
                    for i in 0..<inDim { acc += wp[base + i] * xp[i] }
                    y[o] = relu ? max(acc, 0) : acc
                }
            }
        }
        return y
    }

    func probs(_ features: [Double]) -> [Float] {
        var x = [Float](repeating: 0, count: mean.count)
        for i in 0..<min(features.count, mean.count) {
            x[i] = (Float(features[i]) - mean[i]) / std[i]
        }
        var total = [Float](repeating: 0, count: classes.count)
        for L in layers {
            let h0 = matvec(L.w0, x, L.b0, L.h0, L.nIn, relu: true)
            let h1 = matvec(L.w1, h0, L.b1, L.h1, L.h0, relu: true)
            let logits = matvec(L.w2, h1, L.b2, L.nOut, L.h1, relu: false)
            let mx = logits.max() ?? 0
            var e = logits.map { expf($0 - mx) }
            let sum = e.reduce(0, +)
            if sum > 0 { for i in 0..<e.count { e[i] /= sum } }
            for i in 0..<total.count { total[i] += e[i] }
        }
        return total.map { $0 / Float(layers.count) }
    }
}

/// Live preview of the phone's own camera.
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }

    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView()
        v.previewLayer.session = session
        v.previewLayer.videoGravity = .resizeAspectFill
        v.backgroundColor = .black
        return v
    }

    func updateUIView(_ v: PreviewView, context: Context) {
        if v.previewLayer.session !== session { v.previewLayer.session = session }
    }
}


/// Latest frame from the glasses, for the preview only.
final class GlassesFeed: ObservableObject {
    @Published var image: UIImage?
}

/// Reads the glasses' /camera MJPEG stream and hands over the newest complete
/// JPEG in each chunk received; older ones are dropped rather than queued, so
/// recognition never falls behind real time.
final class GlassesStream: NSObject, URLSessionDataDelegate {
    var onJPEG: ((Data) -> Void)?
    var onEnd: (() -> Void)?
    private var session: URLSession?
    private var buffer = Data()
    private let soi = Data([0xFF, 0xD8]), eoi = Data([0xFF, 0xD9])

    func start(_ url: URL) {
        stop()
        buffer.removeAll()
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 10
        session = URLSession(configuration: config, delegate: self, delegateQueue: nil)
        session?.dataTask(with: url).resume()
    }

    func stop() {
        session?.invalidateAndCancel()
        session = nil
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        buffer.append(data)
        if buffer.count > 4_000_000 { buffer.removeAll(); return }
        var newest: Data?
        while let start = buffer.range(of: soi),
              let end = buffer.range(of: eoi, in: start.upperBound..<buffer.endIndex) {
            newest = buffer.subdata(in: start.lowerBound..<end.upperBound)
            buffer.removeSubrange(buffer.startIndex..<end.upperBound)
        }
        if let newest { onJPEG?(newest) }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard session === self.session else { return }   // ignore a replaced session
        onEnd?()
    }
}

/// The glasses' video, drawn from the same frames recognition runs on.
struct GlassesPreview: View {
    @ObservedObject var feed: GlassesFeed
    var body: some View {
        if let image = feed.image {
            Image(uiImage: image).resizable().scaledToFill()
        } else {
            Color.black
        }
    }
}
