import Foundation
import NaturalLanguage

/// English word order -> ASL word order.
///
/// Substituting signs for English words one at a time does not produce ASL. It
/// produces Signed Exact English, which is a different thing and reads to a
/// fluent signer roughly the way word-for-word Japanese reads to an English
/// speaker. ASL has its own syntax, and the differences are systematic enough
/// to encode:
///
///   * TIME FIRST. Temporal context is established before anything else.
///     "I went to the store yesterday" -> YESTERDAY I GO STORE
///   * TENSE ONCE. There is no verb inflection. Past is marked with FINISH and
///     future with WILL, stated once, then dropped.
///     "I ate and I drank" -> FINISH I EAT I DRINK
///   * WH-WORD LAST. Question words go to the end, not the front.
///     "What is your name?" -> YOUR NAME WHAT
///   * NEGATION AFTER THE VERB.  "I don't know" -> I KNOW NOT
///   * ADJECTIVE AFTER NOUN.  "the blue car" -> CAR BLUE
///   * NO ARTICLES, NO COPULA. Handled in the word map.
///
/// Part of speech comes from NLTagger, which ships with iOS and runs offline,
/// so this costs nothing at runtime and works with no network.
///
/// HONEST SCOPE: this is a rule-based approximation, not translation.
/// English-to-ASL is an open research problem, and the parts this cannot do are
/// the parts that live off the hands entirely — spatial agreement (assigning
/// referents to locations and pointing back at them), classifier predicates,
/// role shift, and the grammatical facial expressions that mark topics,
/// conditionals and question type. What it does deliver is correct constituent
/// ORDER for common sentences, which is the difference between "obviously a
/// word-for-word machine" and "recognisably ASL".
enum ASLGrammar {

    /// Temporal expressions, which move to the front of the sentence.
    private static let timeWords: Set<String> = [
        "yesterday", "today", "tomorrow", "tonight", "now", "later", "soon",
        "morning", "afternoon", "evening", "night", "week", "month", "year",
        "day", "hour", "minute", "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday", "always", "sometimes", "often",
        "daily", "weekly", "yearly", "recently", "ago", "already", "yet",
        "before", "after", "while", "during", "past", "future", "next", "last",
    ]

    private static let whWords: Set<String> = [
        "who", "what", "where", "when", "why", "how", "which", "whose", "whom",
    ]

    /// Contractions and negatives, mapped to the sign that carries them.
    private static let negations: [String: String] = [
        "not": "not", "no": "no", "never": "never", "none": "none",
        "dont": "not", "doesnt": "not", "didnt": "not", "cant": "cannot",
        "cannot": "cannot", "wont": "not", "isnt": "not", "arent": "not",
        "wasnt": "not", "werent": "not", "shouldnt": "not", "couldnt": "cannot",
        "wouldnt": "not", "havent": "not", "hasnt": "not", "nothing": "nothing",
    ]

    /// Past-tense auxiliaries and irregular pasts that justify a FINISH marker.
    private static let pastMarkers: Set<String> = [
        "was", "were", "had", "did", "went", "ate", "drank", "saw", "came",
        "took", "made", "said", "got", "gave", "told", "knew", "thought",
        "found", "left", "felt", "bought", "brought", "taught", "met", "paid",
        "sat", "stood", "slept", "wrote", "ran", "began", "chose", "lost",
    ]

    private static let futureMarkers: Set<String> = ["will", "shall", "gonna"]

    /// One English word, with what the tagger made of it.
    private struct Token {
        var text: String
        var pos: NLTag?
        var isTime: Bool
        var isWh: Bool
    }

    /// English sentence -> ASL gloss order.
    ///
    /// Returns lowercase gloss tokens. Lexical choice (which sign carries which
    /// English word) happens afterwards in the word map; this decides only what
    /// order they go in and what gets added or removed.
    static func gloss(_ english: String) -> [String] {
        var tokens = tag(english)
        guard !tokens.isEmpty else { return [] }

        // --- tense, before anything is moved ------------------------------
        var tenseMarker: String?
        if tokens.contains(where: { futureMarkers.contains($0.text) }) {
            tenseMarker = "will"
        } else if tokens.contains(where: { pastMarkers.contains($0.text) })
                    || tokens.contains(where: { isPastParticiple($0) }) {
            tenseMarker = "finish"
        }
        // The auxiliaries themselves carry no sign once the marker exists.
        tokens.removeAll { futureMarkers.contains($0.text) }

        // --- negation -----------------------------------------------------
        // Pulled out and re-inserted after the verb, since ASL negates by
        // following the predicate rather than preceding it.
        var negation: String?
        for (i, t) in tokens.enumerated() where negations[t.text] != nil {
            negation = negations[t.text]
            tokens.remove(at: i)
            break
        }

        // --- extract the pieces that move ---------------------------------
        let time = tokens.filter { $0.isTime }.map(\.text)
        let wh = tokens.filter { $0.isWh }.map(\.text)
        tokens.removeAll { $0.isTime || $0.isWh }

        // --- adjective after noun -----------------------------------------
        tokens = swapAdjectiveNoun(tokens)

        var body = tokens.map(\.text)

        // Negation closes the clause. Inserting it immediately after the verb
        // gives "I HEAR CANNOT YOU", which strands the object; at the end it
        // reads "I HEAR YOU CANNOT", and "I KNOW NOT" is unchanged either way.
        if let negation { body.append(negation) }

        // --- assemble: TIME + TENSE + BODY + WH ---------------------------
        var out: [String] = []
        out.append(contentsOf: time)
        if let tenseMarker { out.append(tenseMarker) }
        out.append(contentsOf: body)
        out.append(contentsOf: wh)
        return out.filter { !$0.isEmpty }
    }

    // MARK: - Internals

    /// Words ASL marks grammatically rather than lexically, so there is no sign
    /// to look for. Dropped here as well as in the word map, so the gloss
    /// preview shown to the user matches what actually plays.
    private static let dropped: Set<String> = [
        "a", "an", "the", "is", "am", "are", "was", "were", "be", "been",
        "being", "do", "does", "did", "of", "to",
    ]

    private static func tag(_ english: String) -> [Token] {
        let tagger = NLTagger(tagSchemes: [.lexicalClass])
        // Apostrophes go first: the tokeniser splits "don't" into "do" + "n't",
        // which matches neither the negation table nor any sign. Stripping them
        // makes it one token, "dont", which the table has.
        let text = english.lowercased().replacingOccurrences(of: "'", with: "")
            .replacingOccurrences(of: "\u{2019}", with: "")
        tagger.string = text

        var out: [Token] = []
        tagger.enumerateTags(in: text.startIndex..<text.endIndex, unit: .word,
                             scheme: .lexicalClass,
                             options: [.omitPunctuation, .omitWhitespace]) { tag, range in
            // Strip apostrophes so "don't" matches the contraction table.
            let word = String(text[range]).filter { $0.isLetter || $0.isNumber }
            if !word.isEmpty, !dropped.contains(word) {
                out.append(Token(text: word, pos: tag,
                                 isTime: timeWords.contains(word),
                                 isWh: whWords.contains(word)))
            }
            return true
        }
        return out
    }

    /// "-ed" on something the tagger called a verb is good enough evidence of
    /// past tense for a marker, without a full morphological analyser.
    private static func isPastParticiple(_ t: Token) -> Bool {
        t.pos == .verb && t.text.hasSuffix("ed") && t.text.count > 3
    }

    /// ASL puts descriptive adjectives after the noun: "CAR BLUE", not
    /// "BLUE CAR". Only adjacent adjective-noun pairs are swapped, since
    /// anything cleverer needs a real parse.
    private static func swapAdjectiveNoun(_ tokens: [Token]) -> [Token] {
        var out = tokens
        var i = 0
        while i < out.count - 1 {
            if out[i].pos == .adjective, out[i + 1].pos == .noun {
                out.swapAt(i, i + 1)
                i += 2                    // do not re-examine the pair
            } else {
                i += 1
            }
        }
        return out
    }
}
