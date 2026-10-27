import Foundation

/// Turns English into a plan of signs, and reports what vocabulary exists.
///
/// It no longer holds any motion data. It used to carry 33 MB of landmark
/// tracks and 26 handshapes to drive the avatar; playback is now video clips
/// and photographs, so all this has left to do is decide which words have a
/// sign and which get spelled.
@MainActor
final class OfflineAvatar: ObservableObject {

    @Published private(set) var ready = false
    @Published private(set) var vocabularyCount = 0


    /// Vocabulary count for the UI. The real answer is how many clips are
    /// bundled, not what the old library listed.
    func loadIndex() {
        guard !ready else { return }
        vocabularyCount = SignVideos.count
        ready = vocabularyCount > 0 || LetterPhotos.count > 0
    }

    /// A word has a sign if a clip exists for it. Everything else is spelled,
    /// which is what a signer does for names and out-of-vocabulary words.
    func hasWord(_ gloss: String) -> Bool { SignVideos.has(gloss) }

    struct Segment {
        let kind: String
        /// The English word, for the caption.
        let label: String
        /// The sign to play, which is not always the English word: "went" plays
        /// GO, "automobile" plays CAR.
        let gloss: String
        let letters: [String]

        init(kind: String, label: String, gloss: String? = nil, letters: [String]) {
            self.kind = kind
            self.label = label
            self.gloss = gloss ?? label
            self.letters = letters
        }
    }

    /// English -> ASL vocabulary, built by authoring/build_word_map.py.
    /// `drop` are words ASL has no sign for because it marks them
    /// grammatically — spelling T-H-E is not a fallback, it is wrong.
    private var wordMap: [String: String] = [:]
    private var dropWords: Set<String> = []

    private func loadWordMapIfNeeded() {
        guard wordMap.isEmpty else { return }
        guard let url = Bundle.main.url(forResource: "word_map", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return }
        wordMap = (obj["map"] as? [String: String]) ?? [:]
        dropWords = Set((obj["drop"] as? [String]) ?? [])
    }

    /// The sign for an English word, following inflection and synonymy.
    /// Nil means fingerspell it.
    private func signFor(_ token: String) -> String? {
        if SignVideos.has(token) { return token }
        if let mapped = wordMap[token], SignVideos.has(mapped) { return mapped }
        return nil
    }

    /// English text -> avatar segments, in ASL order.
    ///
    /// Two stages, deliberately separate. ASLGrammar decides ORDER — what moves
    /// to the front, what gets dropped, what gets added — and the word map
    /// decides LEXICON, which sign carries which word. Mixing the two is how
    /// you end up reordering things you then discover you have no sign for.
    func plan(_ text: String) -> (segments: [Segment], unknown: [String]) {
        loadWordMapIfNeeded()
        var segments: [Segment] = []
        var unknown: [String] = []
        let tokens = ASLGrammar.gloss(text).map { Substring($0) }
        var i = 0
        while i < tokens.count {
            let raw = tokens[i]
            i += 1
            let token = raw.lowercased().filter { $0.isLetter || $0.isNumber }
            guard !token.isEmpty else { continue }
            // Some glosses are two English words for one sign ("thank you").
            // Checking the pair first stops THANK YOU being spelled out letter
            // by letter, which is what a one-token-at-a-time loop would do.
            if i < tokens.count {
                let next = tokens[i].lowercased().filter { $0.isLetter || $0.isNumber }
                let pair = token + " " + next
                if !next.isEmpty, hasWord(pair) {
                    segments.append(Segment(kind: "word", label: pair, letters: []))
                    i += 1
                    continue
                }
            }
            // ASL has no articles and no copula. These are not missing signs,
            // they are words that should not appear at all.
            if dropWords.contains(token) { continue }

            if let gloss = signFor(token) {
                segments.append(Segment(kind: "word", label: token,
                                        gloss: gloss, letters: []))
                continue
            }
            let spelled = token.uppercased().compactMap {
                LetterPhotos.has(String($0)) ? String($0) : nil
            }
            if spelled.isEmpty { unknown.append(String(raw)) }
            else { segments.append(Segment(kind: "fingerspell", label: String(raw), letters: spelled)) }
        }
        return (segments, unknown)
    }
}
