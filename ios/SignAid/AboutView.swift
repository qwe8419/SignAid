import SwiftUI

/// How SignAid works, what it does with your data, and who made the material it
/// is built on. The CC BY datasets require attribution; this is where it lives.
struct AboutView: View {
    var body: some View {
        Form {
            Section {
                VStack(alignment: .leading, spacing: 8) {
                    Text("SignAid").font(.title2.bold())
                    Text("Two-way ASL ↔ English conversation on one phone.")
                        .foregroundStyle(.secondary)
                    Text("Version \(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "1.0")")
                        .font(.caption).foregroundStyle(.secondary)
                }
                .padding(.vertical, 4)
            }

            Section("Privacy") {
                Text("Recognition, speech and signing all run on this phone. SignAid has no account and no analytics, and camera frames are never stored or sent anywhere. The only network use is the optional SignAid Glasses prototype, which talks to your own computer.")
                    .font(.callout)
            }

            Section("How it works") {
                Text("""
                Google's MediaPipe finds 21 points on your hand in every camera frame. SignAid turns them into 300 measurements of finger angles, distances and direction, and a small neural network trained on over 33,000 examples from 11 signers decides which letter it is. On signers it has never seen, it reads 94% of frames correctly. A short vote across frames keeps one blurry frame from typing the wrong letter.

                For English to ASL, known words play a recording of a Deaf signer; anything else is fingerspelled.
                """)
                .font(.callout)
            }

            Section("Limits") {
                Text("SignAid reads fingerspelling — the ASL alphabet — not full ASL sentences, facial grammar or two-handed signs. J and Z involve motion and are less reliable. It is a communication aid, not a replacement for an interpreter.")
                    .font(.callout)
            }

            Section("Credits") {
                credit("Word sign videos",
                       "PopSign ASL v1.0 — Starner et al., Georgia Tech and RIT/NTID. Signed by a Deaf adult signer. CC BY 4.0. Clips trimmed, placed on a plain studio backdrop, and downscaled.",
                       "https://signdata.cc.gatech.edu/view/datasets/popsign_v1_0/index.html")
                credit("Alphabet training data",
                       "ASL-HG: American Sign Language Hand Gesture Image Dataset (Mendeley Data, doi:10.17632/j4y5w2c8w9.1). CC BY 4.0. Used to extract hand landmarks.",
                       "https://data.mendeley.com/datasets/j4y5w2c8w9/1")
                credit("Alphabet drawings",
                       "\"Sign language A–Z\" series, Wikimedia Commons. Public domain.",
                       "https://commons.wikimedia.org/wiki/File:Sign_language_A.svg")
                credit("Practice coach voice",
                       "Kokoro-82M by hexgrad, Apache License 2.0. Phrases recorded ahead of time and bundled, so practice works offline.",
                       "https://huggingface.co/hexgrad/Kokoro-82M")
                credit("Hand tracking",
                       "MediaPipe Hand Landmarker, Google. Apache License 2.0.",
                       "https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker")
            }
        }
        .navigationTitle("About")
    }

    private func credit(_ title: String, _ detail: String, _ url: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title).font(.subheadline.weight(.semibold))
            Text(detail).font(.caption).foregroundStyle(.secondary)
            if let u = URL(string: url) {
                Link(url, destination: u).font(.caption2).lineLimit(1)
            }
        }
        .padding(.vertical, 2)
    }
}
