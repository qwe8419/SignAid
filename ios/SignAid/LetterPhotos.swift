import SwiftUI

/// Fingerspelling as line drawings of the alphabet.
///
/// From the "Sign language X.svg" set on Wikimedia Commons, which is complete
/// for A-Z, drawn in a single style, and **public domain** — so unlike the sign
/// videos there is no attribution obligation and nothing to license before
/// shipping. See authoring/fetch_letter_images.py and letter_photos_CREDITS.json.
///
/// Drawings suit this job better than photographs. A fingerspelled letter is a
/// shape, and a drawing strips away the skin tone, lighting and background a
/// reader would otherwise have to look past. J and Z carry motion arrows, which
/// a still photograph cannot show at all — both letters are defined by movement.
enum LetterPhotos {

    /// Bundled as a folder reference so the 26 files keep their own namespace.
    private static let root: URL? =
        Bundle.main.url(forResource: "letter_photos", withExtension: nil)

    private static let available: Set<String> = {
        guard let root,
              let names = try? FileManager.default.contentsOfDirectory(atPath: root.path)
        else { return [] }
        return Set(names.filter { $0.hasSuffix(".jpg") }
                        .map { String($0.dropLast(4)).uppercased() })
    }()

    static var count: Int { available.count }

    static func has(_ letter: String) -> Bool {
        available.contains(letter.uppercased())
    }

    static func url(for letter: String) -> URL? {
        let key = letter.uppercased()
        guard let root, available.contains(key) else { return nil }
        return root.appendingPathComponent(key + ".jpg")
    }

    /// Decoded once and kept. The images are ~14 KB each and a spelled word
    /// revisits letters, so re-decoding per frame would be wasted work.
    private static var cache: [String: UIImage] = [:]

    static func image(for letter: String) -> UIImage? {
        let key = letter.uppercased()
        if let hit = cache[key] { return hit }
        guard let url = url(for: key), let image = UIImage(contentsOfFile: url.path)
        else { return nil }
        cache[key] = image
        return image
    }
}

/// One fingerspelled letter, filling the frame.
struct LetterPhotoView: View {
    var letter: String

    var body: some View {
        ZStack {
            Color(red: 0.10, green: 0.11, blue: 0.14)
            if let image = LetterPhotos.image(for: letter) {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    // The letter itself, large enough to read across a table,
                    // since the point is for the other person to follow along.
                    // Dark ink: the drawings sit on a light card.
                    .overlay(alignment: .topLeading) {
                        Text(letter.uppercased())
                            .font(.system(size: 44, weight: .bold, design: .rounded))
                            .foregroundStyle(Color(white: 0.18))
                            .padding(14)
                    }
            } else {
                Text(letter.uppercased())
                    .font(.system(size: 96, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)
            }
        }
    }
}
