import SwiftUI

/// The main screen, laid out like a translation app because that is what it is:
/// two people who do not share a language, one phone between them.
///
///   ASL -> English : the Deaf/HoH signer signs, text appears, phone speaks it
///   English -> ASL : the hearing person talks, the avatar signs it back
///
/// The swap button flips direction. Each turn lands in a shared transcript so
/// both people can see the conversation so far, which matters when speech is
/// missed or a sign is misread.
struct TranslateView: View {
    @ObservedObject var client: SignAidClient
    @StateObject private var speech = SpeechService()
    @StateObject private var onDevice = OnDeviceRecognizer()
    @StateObject private var offline = OfflineAvatar()
    @StateObject private var speller = OfflineAutocorrect()
    /// Phone-only mode needs no Pi, no Mac and no network.
    @AppStorage("usePhoneCamera") private var usePhoneCamera = true
    /// The glasses source picker only appears once the prototype is switched on
    /// in Settings; everyone else just gets the phone.
    @AppStorage("glassesEnabled") private var glassesEnabled = false
    /// Tuning and data-collection controls. Off for anyone who is not working
    /// on the model.
    @AppStorage("developerMode") private var developerMode = false
    @State private var practiceMode = false
    /// The camera waits for the welcome tour, so its permission prompt arrives
    /// after the person knows what the app is for, not on top of page one.
    @AppStorage("seenIntro") private var seenIntro = false
    // Collection lives INSIDE the phone pane rather than on a pushed screen.
    // A second CameraPreview bound to the same AVCaptureSession was the most
    // likely cause of the freeze, and reusing the one preview that already
    // renders smoothly removes that whole class of problem.
    @State private var collectMode = false
    @State private var collectLetter = "A"
    @State private var collectBurst = 0
    @State private var collectStatus = ""
    @State private var collectCounts: [String: Int] = [:]
    private static let collectVariations = [
        "straight on, normal distance", "rotated slightly left",
        "rotated slightly right", "closer to the camera",
        "further from the camera", "hand higher in frame",
        "hand lower in frame", "tilted toward the camera",
    ]

    @State private var signingToEnglish = true
    @State private var typed = ""
    /// Lets the keyboard be put away: TextEditor treats Return as a newline,
    /// so without this there was no way to close it.
    @FocusState private var typing: Bool
    @State private var turns: [Turn] = []
    @State private var showTranscript = false

    struct Turn: Identifiable {
        let id = UUID()
        let text: String
        let fromSigner: Bool
        let time = Date()
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                directionBar
                Divider().overlay(Color.white.opacity(0.15))

                ScrollView {
                    VStack(spacing: 14) {
                        if signingToEnglish {
                            signerPane
                        } else {
                            hearingPane
                        }
                        if !turns.isEmpty {
                            transcriptSection
                        }
                    }
                    .padding()
                }
                .scrollDismissesKeyboard(.interactively)
            }
            .background(Color.black.ignoresSafeArea())
            .navigationTitle("SignAid")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    NavigationLink { SettingsView(client: client) } label: {
                        Image(systemName: "gearshape.fill")
                    }
                }
            }
            .task {
                offline.loadIndex()
                await speller.load()
            }
        }
    }

    /// Grey placeholder, white for settled text, green for whatever just landed.
    private var highlighted: AttributedString {
        if client.text.isEmpty {
            var s = AttributedString("Sign to start…")
            s.foregroundColor = .secondary
            return s
        }
        var full = AttributedString(client.text)
        full.foregroundColor = .white
        if let added = client.justCommitted, client.text.hasSuffix(added),
           let r = full.range(of: added, options: .backwards) {
            full[r].foregroundColor = .green
        }
        return full
    }

    // MARK: - direction selector (the "language" bar)

    private var directionBar: some View {
        HStack(spacing: 10) {
            languageChip(signingToEnglish ? "ASL" : "English", active: true)
            Button {
                withAnimation(.easeInOut(duration: 0.2)) {
                    signingToEnglish.toggle()
                    speech.stopListening()
                    speech.stopSpeaking()
                }
            } label: {
                Image(systemName: "arrow.left.arrow.right")
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(.green)
                    .padding(8)
                    .background(Circle().fill(Color.white.opacity(0.08)))
            }
            languageChip(signingToEnglish ? "English" : "ASL", active: false)
        }
        .padding(.vertical, 10)
        .frame(maxWidth: .infinity)
        .background(Color(white: 0.09))
    }

    private func languageChip(_ label: String, active: Bool) -> some View {
        Text(label)
            .font(.system(size: 16, weight: .semibold))
            .foregroundStyle(active ? .green : .secondary)
            .frame(maxWidth: .infinity)
    }

    // MARK: - ASL -> English (the signer's turn)

    /// Glasses are the camera only when the prototype is switched on and picked.
    private var usingGlasses: Bool { glassesEnabled && !usePhoneCamera }

    /// One screen for both cameras. With the glasses, the Pi streams hand points
    /// and the phone does everything else with the same code it uses for its
    /// own camera, so typing, Practice and the controls are identical.
    private var signerPane: some View {
        VStack(spacing: 12) {
            if glassesEnabled {
                Picker("Source", selection: $usePhoneCamera) {
                    Text("This phone").tag(true)
                    Text("Glasses").tag(false)
                }
                .pickerStyle(.segmented)
                .onChange(of: usePhoneCamera) { _, _ in startRecognizer() }
            }
            phonePane
        }
        .onAppear { if seenIntro { startRecognizer() } }
        .onChange(of: seenIntro) { _, seen in if seen { startRecognizer() } }
        .onDisappear { onDevice.stop(); onDevice.stopGlasses() }
    }

    private func startRecognizer() {
        guard usingGlasses else {
            onDevice.stopGlasses()
            onDevice.start()
            return
        }
        onDevice.stop()
        Task {
            // Find them first if the saved address does not answer.
            await client.checkHealth()
            if !client.connected { await client.findGlasses() }
            if let url = client.cameraURL { onDevice.startGlasses(cameraURL: url) }
        }
    }

    // MARK: - phone-only recognition

    private var phonePane: some View {
        VStack(spacing: 12) {
            Picker("Mode", selection: $practiceMode) {
                Text("Translate").tag(false)
                Text("Practice").tag(true)
            }
            .pickerStyle(.segmented)
            .disabled(collectMode)

            ZStack(alignment: .topTrailing) {
                // Shorter than the full 3:4 frame so the typed text stays on
                // screen without scrolling; the preview fills and crops.
                Group {
                    if usingGlasses {
                        ZStack {
                            Color.black
                            GlassesPreview(feed: onDevice.glassesFeed)
                            if !onDevice.glassesConnected {
                                VStack(spacing: 6) {
                                    ProgressView().tint(.white)
                                    Text(client.searching ? "Looking for your glasses…"
                                                          : "Connecting to the glasses…")
                                        .font(.caption).foregroundStyle(.secondary)
                                    if !client.searching && !client.connected {
                                        Button("Search again") { startRecognizer() }
                                            .font(.caption).buttonStyle(.bordered)
                                    }
                                }
                            }
                        }
                    } else {
                        CameraPreview(session: onDevice.session)
                    }
                }
                    .frame(maxWidth: .infinity)
                    .frame(height: 330)
                    .clipShape(RoundedRectangle(cornerRadius: 14))
                HStack(spacing: 8) {
                    if developerMode {
                        Text(onDevice.handVisible
                             ? "\(Int(onDevice.fps)) fps" : "no hand")
                            .font(.caption2).padding(6)
                            .background(.black.opacity(0.6))
                            .clipShape(Capsule())
                    }
                    if !usingGlasses {
                    Button { onDevice.flipCamera() } label: {
                        Image(systemName: "arrow.triangle.2.circlepath.camera.fill")
                            .font(.system(size: 15))
                            .padding(8)
                            .background(.black.opacity(0.6))
                            .clipShape(Circle())
                    }
                    .tint(.white)
                    }
                }
                .padding(8)
            }

            if practiceMode {
                PracticePanel(onDevice: onDevice)
            } else {
                translateControls
            }

            if developerMode {
                developerControls
            }

            if !practiceMode {
                Label("Double letter? Slide your hand a little to the side.",
                      systemImage: "arrow.left.and.right")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Label(usingGlasses
                  ? "Glasses camera · recognition runs on this phone"
                  : "Runs entirely on this phone — nothing leaves the device.",
                  systemImage: usingGlasses ? "eyeglasses" : "lock.fill")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var translateControls: some View {
        VStack(spacing: 12) {
            // Tapping the live readout places whatever letter it is showing.
            // The gate stays strict for automatic commits; this is the manual
            // path for a correct read it declined.
            LiveIndicator(label: onDevice.liveLabel,
                          confidence: onDevice.liveConfidence,
                          progress: onDevice.progress, auto: true)
                .contentShape(Rectangle())
                .onTapGesture {
                    if !onDevice.liveLabel.isEmpty { onDevice.append(onDevice.liveLabel) }
                }


            if onDevice.handTooSmall && onDevice.handVisible {
                Label("Move your hand closer — too small to read U/H/R apart",
                      systemImage: "arrow.up.left.and.arrow.down.right")
                    .font(.caption).foregroundStyle(.orange)
            }

            if let err = onDevice.trackerError {
                Text(err).font(.caption).foregroundStyle(.red)
            }
            if onDevice.belowThreshold && onDevice.handVisible {
                Text("Seen, but not confident enough to add — adjust your hand")
                    .font(.caption).foregroundStyle(.orange)
            }

            ZStack(alignment: .topTrailing) {
                Text(onDevice.text.isEmpty ? "Sign to start…" : onDevice.text)
                    .font(.system(size: 26, weight: .medium))
                    .foregroundStyle(onDevice.text.isEmpty ? Color.secondary : Color.white)
                    .frame(maxWidth: .infinity, minHeight: 90, alignment: .topLeading)
                    .padding(14)
                    .background(Color(white: 0.12))
                    .clipShape(RoundedRectangle(cornerRadius: 14))
                    .overlay(RoundedRectangle(cornerRadius: 14)
                        .stroke(.green, lineWidth: onDevice.justCommitted != nil ? 2.5 : 0))
                if let added = onDevice.justCommitted {
                    Text(added)
                        .font(.system(size: 34, weight: .bold)).foregroundStyle(.black)
                        .padding(.horizontal, 16).padding(.vertical, 8)
                        .background(Capsule().fill(.green)).padding(10)
                        .transition(.scale(scale: 0.4).combined(with: .opacity))
                }
            }
            .animation(.spring(response: 0.28, dampingFraction: 0.6),
                       value: onDevice.commitCount)

            if !phoneSuggestions.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack {
                        ForEach(phoneSuggestions, id: \.self) { s in
                            Button(s) { applyPhoneSuggestion(s) }
                                .buttonStyle(.bordered).tint(.orange)
                        }
                    }
                }
            }

            HStack(spacing: 10) {
                smallButton("Space", "space") { onDevice.space() }
                smallButton("Delete", "delete.left") { onDevice.delete() }
                smallButton("Fix", "wand.and.stars") {
                    onDevice.text = speller.correctAll(onDevice.text).0
                }
                smallButton("Clear", "trash") { onDevice.clear() }
            }

            Button {
                let said = onDevice.text.trimmingCharacters(in: .whitespaces)
                guard !said.isEmpty else { return }
                speech.speak(said)
                turns.append(Turn(text: said, fromSigner: true))
                onDevice.clear()
            } label: {
                Label(speech.isSpeaking ? "Speaking…" : "Say it out loud",
                      systemImage: "speaker.wave.2.fill")
                    .frame(maxWidth: .infinity).padding(.vertical, 8)
            }
            .buttonStyle(.borderedProminent).tint(.green)
            .disabled(onDevice.text.trimmingCharacters(in: .whitespaces).isEmpty)


        }
    }

    /// Model tuning and training-data capture. Hidden unless Developer mode is
    /// on in Settings.
    private var developerControls: some View {
        VStack(spacing: 12) {
            Divider().overlay(Color.orange.opacity(0.4))
            Text("Developer").font(.caption.weight(.semibold)).foregroundStyle(.orange)
            Toggle("Flip hand direction (try if H and U swap)", isOn: $onDevice.mirrorLandmarks)
                .toggleStyle(.switch).tint(.green).font(.caption)

            HStack(spacing: 10) {
                Toggle("Image model", isOn: $onDevice.useImageModel)
                    .toggleStyle(.switch).tint(.orange)
                    .onChange(of: onDevice.useImageModel) { _, on in
                        onDevice.wantsDebugCrop = on
                    }
                if let crop = onDevice.debugCrop {
                    Image(uiImage: crop)
                        .resizable().scaledToFill()
                        .frame(width: 54, height: 54)
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                        .overlay(RoundedRectangle(cornerRadius: 6)
                            .stroke(.orange, lineWidth: 1))
                }
                VStack(alignment: .leading, spacing: 1) {
                    Text("image model sees").font(.caption2).foregroundStyle(.secondary)
                    Text(onDevice.cropLabel).font(.caption).foregroundStyle(.orange)
                    Text(onDevice.fusionState).font(.caption2)
                        .foregroundStyle(onDevice.fusionState == "fused" ? .green : .secondary)
                }
                Spacer()
            }
            .font(.caption)

            Button {
                collectMode.toggle()
                onDevice.autoCommit = !collectMode
                if !collectMode { onDevice.stopCollecting() }
                if collectMode { Task { await refreshCollectCounts() } }
            } label: {
                Label(collectMode ? "Done collecting" : "Collect training data",
                      systemImage: "plus.viewfinder")
                    .font(.caption)
            }
            .buttonStyle(.bordered)
            .tint(collectMode ? .red : .gray)

            if collectMode { collectPanel }
        }
    }

    /// Suggestions for the word being fingerspelled, computed on-device.
    private var phoneSuggestions: [String] {
        guard let last = onDevice.text.split(separator: " ").last else { return [] }
        return speller.correct(String(last)).1
    }

    private func applyPhoneSuggestion(_ s: String) {
        var words = onDevice.text.split(separator: " ").map(String.init)
        guard !words.isEmpty else { return }
        words[words.count - 1] = s
        onDevice.text = words.joined(separator: " ") + " "
    }

    private var collectPanel: some View {
        VStack(spacing: 10) {
            Text("Next burst: \(Self.collectVariations[collectBurst % Self.collectVariations.count])")
                .font(.caption).foregroundStyle(.orange)
                .multilineTextAlignment(.center)

            HStack(spacing: 12) {
                Menu {
                    ForEach(0..<26, id: \.self) { i in
                        let l = String(UnicodeScalar(65 + i)!)
                        Button(l) { collectLetter = l }
                    }
                } label: {
                    Text(collectLetter)
                        .font(.system(size: 28, weight: .bold))
                        .frame(width: 66, height: 48)
                        .background(Color(white: 0.16))
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                }
                Button {
                    if let u = collectLetter.unicodeScalars.first, u.value < 90 {
                        collectLetter = String(UnicodeScalar(u.value + 1)!)
                    }
                } label: {
                    Image(systemName: "chevron.right.circle.fill").font(.title2)
                }
                Spacer()
                Text("\(collectCounts[collectLetter] ?? 0) saved")
                    .font(.caption).foregroundStyle(.secondary)
            }

            if onDevice.collecting {
                ProgressView(value: Double(onDevice.collectedCount), total: 25).tint(.red)
            }

            Button {
                if onDevice.collecting { finishCollectBurst() }
                else { onDevice.startCollecting(target: 25) }
            } label: {
                Label(onDevice.collecting
                        ? "Recording \(onDevice.collectedCount)/25"
                        : "Record burst of 25",
                      systemImage: onDevice.collecting ? "stop.circle.fill" : "record.circle")
                    .frame(maxWidth: .infinity).padding(.vertical, 6)
            }
            .buttonStyle(.borderedProminent)
            .tint(onDevice.collecting ? .red : .green)

            Text(collectStatus).font(.caption).foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(10)
        .background(Color(white: 0.11))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .onChange(of: onDevice.collecting) { was, now in
            if was && !now { finishCollectBurst() }
        }
    }

    private func finishCollectBurst() {
        let payload = onDevice.stopCollecting()
        guard payload.count >= 5 else {
            collectStatus = "Only \(payload.count) frames — keep your hand in view"
            return
        }
        let l = collectLetter, b = collectBurst
        let v = Self.collectVariations[collectBurst % Self.collectVariations.count]
        collectBurst += 1
        // Saved on the phone first, so recording works with no server running.
        // The upload is a convenience on top, not the thing it depends on.
        let saved = onDevice.saveBurst(letter: l, burst: b, variation: v,
                                       samples: payload)
        collectStatus = saved ? "Saved \(payload.count) for \(l)"
                              : "Could not write to disk"
        Task { await refreshCollectCounts() }
        Task { _ = await client.uploadSamples(letter: l, burst: b,
                                              variation: v, samples: payload) }
    }

    private func refreshCollectCounts() async {
        // On-disk counts, not the server's: the phone is the source of truth.
        collectCounts = OnDeviceRecognizer.savedCounts()
    }

    private var serverPane: some View {
        VStack(spacing: 12) {
            if client.connected {
                MJPEGView(url: client.videoURL)
                    .aspectRatio(4/3, contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 14))
                    .overlay(alignment: .topLeading) {
                        if client.autoOn {
                            Label("Live", systemImage: "dot.radiowaves.left.and.right")
                                .font(.caption2).padding(6)
                                .background(.green.opacity(0.85))
                                .foregroundStyle(.black)
                                .clipShape(Capsule())
                                .padding(8)
                        }
                    }
            } else {
                DisconnectedCard(client: client)
            }

            // Always shown, not just in continuous mode: knowing whether the
            // camera can see your hand matters just as much when tapping
            // Detect by hand.
            if client.connected {
                LiveIndicator(label: client.liveLabel,
                              confidence: client.liveConfidence,
                              progress: client.autoOn ? client.autoProgress : 0,
                              auto: client.autoOn)
            }

            // What the signer has produced. The newest characters are tinted so
            // it is obvious which part just arrived.
            ZStack(alignment: .topTrailing) {
                Text(highlighted)
                    .font(.system(size: 26, weight: .medium))
                    .frame(maxWidth: .infinity, minHeight: 90, alignment: .topLeading)
                    .padding(14)
                    .background(Color(white: 0.12))
                    .clipShape(RoundedRectangle(cornerRadius: 14))
                    .overlay(
                        RoundedRectangle(cornerRadius: 14)
                            .stroke(.green, lineWidth: client.justCommitted != nil ? 2.5 : 0)
                    )
                    .animation(.easeOut(duration: 0.18), value: client.commitCount)

                if let added = client.justCommitted {
                    Text(added)
                        .font(.system(size: 34, weight: .bold))
                        .foregroundStyle(.black)
                        .padding(.horizontal, 16).padding(.vertical, 8)
                        .background(Capsule().fill(.green))
                        .padding(10)
                        .transition(.scale(scale: 0.4).combined(with: .opacity))
                }
            }
            .animation(.spring(response: 0.28, dampingFraction: 0.6),
                       value: client.commitCount)

            if !client.suggestions.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack {
                        ForEach(client.suggestions, id: \.self) { s in
                            Button(s) { Task { await client.apply(suggestion: s) } }
                                .buttonStyle(.bordered).tint(.orange)
                        }
                    }
                }
            }

            Toggle("Detect continuously", isOn: Binding(
                get: { client.autoOn },
                set: { on in Task { await client.setAuto(on) } }))
                .tint(.green)

            HStack(spacing: 10) {
                smallButton("Letter", "hand.raised") { await client.detectLetter() }
                smallButton("Word", "text.word.spacing") { await client.detectWord() }
                smallButton("Space", "space") { await client.addSpace() }
                smallButton("Delete", "delete.left") { await client.deleteLast() }
            }

            // The action that matters: say it out loud to the hearing person.
            Button {
                let said = client.text.trimmingCharacters(in: .whitespaces)
                guard !said.isEmpty else { return }
                speech.speak(said)
                turns.append(Turn(text: said, fromSigner: true))
                Task { await client.clearText() }
            } label: {
                Label(speech.isSpeaking ? "Speaking…" : "Say it out loud",
                      systemImage: speech.isSpeaking ? "speaker.wave.3.fill" : "speaker.wave.2.fill")
                    .frame(maxWidth: .infinity).padding(.vertical, 8)
            }
            .buttonStyle(.borderedProminent).tint(.green)
            .disabled(client.text.trimmingCharacters(in: .whitespaces).isEmpty)

            Button("Clear") { Task { await client.clearText() } }
                .buttonStyle(.bordered).tint(.gray)

            Text(client.message).font(.caption).foregroundStyle(.secondary)
        }
    }

    // MARK: - English -> ASL (the hearing person's turn)

    private var hearingPane: some View {
        VStack(spacing: 12) {
            AvatarPlayer(client: client, offline: offline, text: $typed)
                .aspectRatio(4/5, contentMode: .fit)
                .clipShape(RoundedRectangle(cornerRadius: 14))

            ZStack(alignment: .topLeading) {
                if typed.isEmpty && speech.transcript.isEmpty {
                    Text("Tap the mic and speak, or type…")
                        .foregroundStyle(.secondary).padding(18)
                }
                TextEditor(text: Binding(
                    get: { speech.isListening ? speech.transcript : typed },
                    set: { typed = $0 }))
                    .scrollContentBackground(.hidden)
                    .frame(minHeight: 90)
                    .padding(8)
                    .font(.system(size: 22))
                    .focused($typing)
                    .toolbar {
                        ToolbarItemGroup(placement: .keyboard) {
                            Spacer()
                            Button("Done") { typing = false }.bold()
                        }
                    }
            }
            .background(Color(white: 0.12))
            .clipShape(RoundedRectangle(cornerRadius: 14))

            HStack(spacing: 12) {
                Button {
                    typing = false
                    if speech.isListening {
                        speech.stopListening()
                        if !speech.transcript.isEmpty { typed = speech.transcript }
                    } else {
                        speech.startListening()
                    }
                } label: {
                    Label(speech.isListening ? "Stop" : "Speak",
                          systemImage: speech.isListening ? "stop.circle.fill" : "mic.fill")
                        .frame(maxWidth: .infinity).padding(.vertical, 8)
                }
                .buttonStyle(.borderedProminent)
                .tint(speech.isListening ? .red : .blue)

                Button {
                    let text = (speech.isListening ? speech.transcript : typed)
                        .trimmingCharacters(in: .whitespaces)
                    guard !text.isEmpty else { return }
                    typing = false
                    speech.stopListening()
                    typed = text
                    turns.append(Turn(text: text, fromSigner: false))
                    NotificationCenter.default.post(name: .signNow, object: text)
                } label: {
                    Label("Sign it", systemImage: "hands.and.sparkles.fill")
                        .frame(maxWidth: .infinity).padding(.vertical, 8)
                }
                .buttonStyle(.borderedProminent).tint(.green)
            }

            if let err = speech.error {
                Text(err).font(.caption).foregroundStyle(.red)
            }
            Text("Known words are signed; names and unknown words are fingerspelled.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    // MARK: - shared transcript

    private var transcriptSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Conversation").font(.headline)
                Spacer()
                Button(showTranscript ? "Hide" : "Show") {
                    withAnimation { showTranscript.toggle() }
                }.font(.subheadline)
                Button("Clear") { turns.removeAll() }
                    .font(.subheadline).tint(.red)
            }
            if showTranscript {
                ForEach(turns.reversed()) { turn in
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: turn.fromSigner ? "hand.raised.fill" : "person.wave.2.fill")
                            .foregroundStyle(turn.fromSigner ? .green : .blue)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(turn.fromSigner ? "Signer" : "Speaker")
                                .font(.caption2).foregroundStyle(.secondary)
                            Text(turn.text).font(.body)
                        }
                        Spacer()
                    }
                    .padding(10)
                    .background(Color(white: 0.1))
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                }
            }
        }
        .padding(.top, 6)
    }

    private func smallButton(_ title: String, _ icon: String,
                             action: @escaping () -> Void) -> some View {
        Button { action() } label: {
            VStack(spacing: 2) {
                Image(systemName: icon)
                Text(title).font(.caption2)
            }
            .frame(maxWidth: .infinity).padding(.vertical, 6)
        }
        .buttonStyle(.bordered).tint(.gray)
    }

    private func smallButton(_ title: String, _ icon: String,
                             action: @escaping () async -> Void) -> some View {
        Button { Task { await action() } } label: {
            VStack(spacing: 2) {
                Image(systemName: icon)
                Text(title).font(.caption2)
            }
            .frame(maxWidth: .infinity).padding(.vertical, 6)
        }
        .buttonStyle(.bordered).tint(.gray)
    }
}

extension Notification.Name {
    static let signNow = Notification.Name("SignAid.signNow")
}


/// Shows what the recogniser is seeing right now and how close it is to
/// committing, so a sign that is nearly-but-not-quite registered is visible
/// rather than silent.
struct LiveIndicator: View {
    let label: String
    let confidence: Double
    let progress: Double
    var auto: Bool = true

    var body: some View {
        HStack(spacing: 14) {
            ZStack {
                Circle().stroke(Color.white.opacity(0.12), lineWidth: 6)
                Circle()
                    .trim(from: 0, to: max(progress, 0.001))
                    .stroke(progress >= 1 ? Color.green : Color.orange,
                            style: StrokeStyle(lineWidth: 6, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                    .animation(.linear(duration: 0.12), value: progress)
                Text(label.isEmpty ? "–" : label)
                    .font(.system(size: 22, weight: .bold))
                    .foregroundStyle(label.isEmpty ? Color.secondary : Color.white)
            }
            .frame(width: 58, height: 58)

            VStack(alignment: .leading, spacing: 3) {
                Text(label.isEmpty ? "Looking for a hand…" : "Reading \(label)")
                    .font(.callout.weight(.medium))
                    .foregroundStyle(label.isEmpty ? Color.secondary : Color.white)
                Text(label.isEmpty ? "Hold your hand in view"
                     : "\(Int(confidence * 100))% sure — tap to add it now")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding(12)
        .background(Color(white: 0.1))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}
