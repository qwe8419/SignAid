import AVFoundation
import Speech
import SwiftUI

/// Speech in and out for the hearing side of the conversation.
///
/// Both run on-device rather than through the server: the hearing person is
/// holding the phone, so their microphone and speaker are right there, and
/// routing audio over wifi to the Pi would add latency and a failure mode for
/// no benefit.
@MainActor
final class SpeechService: NSObject, ObservableObject {

    @Published var transcript = ""
    @Published var isListening = false
    @Published var isSpeaking = false
    @Published var authorized = false
    @Published var error: String?

    private let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private let engine = AVAudioEngine()
    private let synthesizer = AVSpeechSynthesizer()

    override init() {
        super.init()
        synthesizer.delegate = self
    }

    func requestPermission() {
        SFSpeechRecognizer.requestAuthorization { status in
            Task { @MainActor in
                self.authorized = (status == .authorized)
                if status == .denied {
                    self.error = "Speech recognition denied. Enable it in Settings."
                }
            }
        }
        AVAudioApplication.requestRecordPermission { _ in }
    }

    func startListening() {
        guard !isListening else { return }
        // Ask on first use rather than at launch, so the prompt arrives when
        // the person has just tapped Speak and knows why it is being asked.
        if SFSpeechRecognizer.authorizationStatus() == .notDetermined {
            SFSpeechRecognizer.requestAuthorization { status in
                Task { @MainActor in
                    self.authorized = (status == .authorized)
                    if status == .authorized {
                        AVAudioApplication.requestRecordPermission { granted in
                            Task { @MainActor in if granted { self.startListening() } }
                        }
                    } else {
                        self.error = "Speech recognition is off. Turn it on in Settings, or type instead."
                    }
                }
            }
            return
        }
        guard let recognizer, recognizer.isAvailable else {
            error = "Speech recognition unavailable"
            return
        }
        transcript = ""
        error = nil

        do {
            let session = AVAudioSession.sharedInstance()
            // .duckOthers so our own speech output doesn't fight the mic
            try session.setCategory(.playAndRecord, mode: .measurement,
                                    options: [.duckOthers, .defaultToSpeaker])
            try session.setActive(true, options: .notifyOthersOnDeactivation)

            let req = SFSpeechAudioBufferRecognitionRequest()
            req.shouldReportPartialResults = true      // live text as they speak
            // Keep audio on the phone. Without this, Apple's recogniser may send
            // it to Apple's servers, which would make the app's "nothing leaves
            // the device" promise untrue -- and it needs a network connection.
            // Every iPhone that runs iOS 17 supports on-device English.
            if recognizer.supportsOnDeviceRecognition {
                req.requiresOnDeviceRecognition = true
            }
            request = req

            let input = engine.inputNode
            input.removeTap(onBus: 0)
            input.installTap(onBus: 0, bufferSize: 1024,
                             format: input.outputFormat(forBus: 0)) { buffer, _ in
                req.append(buffer)
            }
            engine.prepare()
            try engine.start()
            isListening = true

            task = recognizer.recognitionTask(with: req) { [weak self] result, err in
                Task { @MainActor in
                    guard let self else { return }
                    if let result {
                        self.transcript = result.bestTranscription.formattedString
                    }
                    if err != nil || (result?.isFinal ?? false) {
                        self.stopListening()
                    }
                }
            }
        } catch {
            self.error = "Could not start the microphone"
            isListening = false
        }
    }

    func stopListening() {
        guard isListening || engine.isRunning else { return }
        engine.stop()
        engine.inputNode.removeTap(onBus: 0)
        request?.endAudio()
        task?.cancel()
        request = nil
        task = nil
        isListening = false
    }

    /// Speaks the signer's text aloud for the hearing person.
    func speak(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        stopListening()
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
            try session.setActive(true)
        } catch { /* speaking is best-effort */ }

        let utterance = AVSpeechUtterance(string: trimmed)
        utterance.voice = Self.bestVoice()
        // Premium neural voices sound natural at the default pace; the old
        // compact voice needed slowing down to be understood.
        utterance.rate = Self.bestVoice().quality == .default ? 0.48
                                                              : AVSpeechUtteranceDefaultSpeechRate
        isSpeaking = true
        synthesizer.speak(utterance)
    }

    /// The most natural English voice on this iPhone. Translation has to speak
    /// whatever was signed, so it cannot use the coach's pre-recorded clips;
    /// iOS's Premium voices are the closest on-device match. Apps can use them
    /// but not download them, so Settings explains how (Speaking voice).
    static func bestVoice() -> AVSpeechSynthesisVoice {
        let preferred = ["Zoe", "Ava", "Evan", "Nathan", "Joelle", "Noelle", "Samantha"]
        let english = AVSpeechSynthesisVoice.speechVoices().filter {
            $0.language.hasPrefix("en") && !$0.voiceTraits.contains(.isNoveltyVoice)
        }
        func score(_ v: AVSpeechSynthesisVoice) -> Int {
            let q = v.quality == .premium ? 300 : v.quality == .enhanced ? 200 : 100
            let us = v.language == "en-US" ? 20 : 0
            let rank = preferred.firstIndex { v.name.hasPrefix($0) }.map { 10 - $0 } ?? 0
            return q + us + rank
        }
        return english.max { score($0) < score($1) }
            ?? AVSpeechSynthesisVoice(language: "en-US")!
    }

    static var voiceDescription: String {
        let v = bestVoice()
        let q = v.quality == .premium ? "Premium" : v.quality == .enhanced ? "Enhanced" : "Standard"
        return "\(v.name) (\(q))"
    }

    static var hasNaturalVoice: Bool { bestVoice().quality != .default }

    private static let previewSynth = AVSpeechSynthesizer()
    static func preview() {
        let u = AVSpeechUtterance(string: "Hi! This is how SignAid will sound when it speaks for you.")
        u.voice = bestVoice()
        previewSynth.stopSpeaking(at: .immediate)
        previewSynth.speak(u)
    }

    func stopSpeaking() {
        synthesizer.stopSpeaking(at: .immediate)
        isSpeaking = false
    }
}

extension SpeechService: AVSpeechSynthesizerDelegate {
    nonisolated func speechSynthesizer(_ s: AVSpeechSynthesizer,
                                       didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in self.isSpeaking = false }
    }
    nonisolated func speechSynthesizer(_ s: AVSpeechSynthesizer,
                                       didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor in self.isSpeaking = false }
    }
}
