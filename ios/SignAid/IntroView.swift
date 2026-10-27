import SwiftUI

/// First-launch walkthrough: what the app does, how to hold your hand, and that
/// nothing leaves the phone. Three pages, skippable, shown once.
struct IntroView: View {
    var onDone: () -> Void
    @State private var page = 0

    private struct Page {
        let icon: String
        let title: String
        let body: String
    }

    private let pages = [
        Page(icon: "hand.raised.fill",
             title: "Sign, and it speaks",
             body: "Fingerspell to the camera and SignAid types each letter, suggests the word, and says it out loud for the person you're talking to."),
        Page(icon: "person.wave.2.fill",
             title: "Talk, and it signs",
             body: "Swap direction and the hearing person speaks or types. SignAid shows real Deaf signers signing each word, and fingerspells names."),
        Page(icon: "lock.shield.fill",
             title: "Private by design",
             body: "Everything runs on this phone. No account, no internet, no video ever leaves the device.\n\nTip: hold your hand about an arm's length away, in good light."),
    ]

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Spacer()
                Button("Skip", action: onDone).foregroundStyle(.secondary).padding()
            }
            TabView(selection: $page) {
                ForEach(pages.indices, id: \.self) { i in
                    VStack(spacing: 22) {
                        Spacer()
                        Image(systemName: pages[i].icon)
                            .font(.system(size: 72, weight: .semibold))
                            .foregroundStyle(.green)
                            .frame(height: 96)
                        Text(pages[i].title)
                            .font(.system(size: 30, weight: .bold, design: .rounded))
                            .multilineTextAlignment(.center)
                        Text(pages[i].body)
                            .font(.body).foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 28)
                        Spacer()
                        Spacer()
                    }
                    .tag(i)
                }
            }
            .tabViewStyle(.page(indexDisplayMode: .always))

            Button {
                if page < pages.count - 1 {
                    withAnimation { page += 1 }
                } else {
                    onDone()
                }
            } label: {
                Text(page < pages.count - 1 ? "Next" : "Get started")
                    .font(.headline)
                    .frame(maxWidth: .infinity).padding(.vertical, 14)
            }
            .buttonStyle(.borderedProminent).tint(.green)
            .foregroundStyle(.black)
            .padding(.horizontal, 24).padding(.bottom, 24)
        }
        .background(Color.black.ignoresSafeArea())
        .preferredColorScheme(.dark)
    }
}
