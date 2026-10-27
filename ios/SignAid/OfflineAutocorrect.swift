import Foundation

/// Fingerspelling autocorrect, on-device.
///
/// Edit distance alone picks absurd candidates — a full dictionary will happily
/// turn "helo" into "halo" — so candidates are ranked by how common the word
/// actually is. Ships a compact frequency table rather than depending on a
/// server.
@MainActor
final class OfflineAutocorrect: ObservableObject {

    private var freq: [String: Double] = [:]
    private let alphabet = Array("abcdefghijklmnopqrstuvwxyz")
    /// One edit costs this much zipf frequency. Swept against a fingerspelling
    /// error set: 1.2 let very common two-edit words win ("that" over "thank");
    /// everything >= 1.8 behaved.
    private let editCost = 2.0
    /// Words below this frequency are still corrected even if they exist — the
    /// corpus contains frequent misspellings ("teh" scores 3.04).
    private let keepMinFreq = 3.5

    func load() async {
        guard freq.isEmpty else { return }
        let parsed: [String: Double] = await Task.detached(priority: .utility) {
            guard let url = Bundle.main.url(forResource: "wordfreq_compact", withExtension: "json"),
                  let data = try? Data(contentsOf: url),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Double]
            else { return [:] }
            return obj
        }.value
        freq = parsed
    }

    private func edits1(_ w: String) -> Set<String> {
        var out = Set<String>()
        let chars = Array(w)
        for i in 0...chars.count {
            if i < chars.count {                                  // delete
                var d = chars; d.remove(at: i); out.insert(String(d))
            }
            if i < chars.count - 1 {                              // transpose
                var t = chars; t.swapAt(i, i + 1); out.insert(String(t))
            }
            for c in alphabet {
                if i < chars.count {                              // replace
                    var r = chars; r[i] = c; out.insert(String(r))
                }
                var ins = chars; ins.insert(c, at: i); out.insert(String(ins))   // insert
            }
        }
        return out
    }

    /// -> (best, up to 4 suggestions)
    func correct(_ word: String) -> (String, [String]) {
        let w = word.lowercased()
        guard !w.isEmpty, !freq.isEmpty else { return (word, []) }
        if let f = freq[w], f >= keepMinFreq { return (word, []) }

        var distance: [String: Double] = [:]
        for c in edits1(w) where freq[c] != nil { distance[c] = 1 }
        if distance.count < 3 {
            outer: for e in edits1(w) {
                for c in edits1(e) where freq[c] != nil {
                    if distance[c] == nil { distance[c] = 2 }
                    if distance.count > 400 { break outer }
                }
            }
        }
        guard !distance.isEmpty else { return (word, []) }

        let ranked = distance.keys.sorted {
            (freq[$0]! - editCost * distance[$0]!) > (freq[$1]! - editCost * distance[$1]!)
        }
        let top = Array(ranked.prefix(4))
        if let f = freq[w], let best = top.first, f >= (freq[best] ?? 0) - 0.5 {
            return (word, top)
        }
        return (top.first ?? word, top)
    }

    func correctAll(_ text: String) -> (String, [String]) {
        var out: [String] = []
        var changes: [String] = []
        for w in text.split(separator: " ") {
            let (fixed, _) = correct(String(w))
            if fixed.lowercased() != w.lowercased() { changes.append("\(w)→\(fixed)") }
            out.append(fixed)
        }
        return (out.joined(separator: " ") + (text.hasSuffix(" ") ? " " : ""), changes)
    }
}
