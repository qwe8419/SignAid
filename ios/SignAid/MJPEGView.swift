import SwiftUI
import UIKit

/// Displays the server's /video MJPEG stream.
///
/// AVPlayer and WKWebView both handle MJPEG badly (no seeking, no duration), so
/// this parses the multipart stream directly: find each JPEG's SOI/EOI markers
/// and hand the bytes to UIImage.
final class MJPEGStreamView: UIView {

    private let imageView = UIImageView()
    private var task: URLSessionDataTask?
    private var session: URLSession?
    private var buffer = Data()
    private var decoding = false
    private let decodeQueue = DispatchQueue(label: "signaid.mjpeg.decode",
                                            qos: .userInteractive)
    private let delegateBox = DelegateBox()

    var onFrame: ((Bool) -> Void)?

    override init(frame: CGRect) {
        super.init(frame: frame)
        imageView.contentMode = .scaleAspectFit
        imageView.backgroundColor = .black
        addSubview(imageView)
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) not implemented") }

    override func layoutSubviews() {
        super.layoutSubviews()
        imageView.frame = bounds
    }

    func start(url: URL) {
        stop()
        buffer.removeAll()
        delegateBox.onData = { [weak self] data in self?.consume(data) }
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 15
        let s = URLSession(configuration: config, delegate: delegateBox, delegateQueue: nil)
        session = s
        task = s.dataTask(with: url)
        task?.resume()
    }

    func stop() {
        task?.cancel(); task = nil
        session?.invalidateAndCancel(); session = nil
    }

    private func consume(_ data: Data) {
        buffer.append(data)
        // Cap the buffer so a malformed stream cannot grow it without bound.
        if buffer.count > 4_000_000 { buffer.removeAll() ; return }

        let soi = Data([0xFF, 0xD8])      // JPEG start
        let eoi = Data([0xFF, 0xD9])      // JPEG end

        // Decode only the NEWEST complete frame in whatever arrived. Decoding
        // every one costs CPU we do not have and, worse, walks the display
        // steadily further behind real time -- which is what made signing feel
        // laggy. A dropped frame is invisible; a late frame is not.
        var newest: Data?
        while let start = buffer.range(of: soi),
              let end = buffer.range(of: eoi, in: start.upperBound..<buffer.endIndex) {
            newest = buffer.subdata(in: start.lowerBound..<end.upperBound)
            buffer.removeSubrange(buffer.startIndex..<end.upperBound)
        }
        guard let jpeg = newest else { return }

        // Skip a decode if the previous one has not been drawn yet, so a slow
        // main thread cannot build a backlog.
        guard !decoding else { return }
        decoding = true
        decodeQueue.async { [weak self] in
            let image = UIImage(data: jpeg)
            DispatchQueue.main.async {
                if let image { self?.imageView.image = image; self?.onFrame?(true) }
                self?.decoding = false
            }
        }
    }

    private final class DelegateBox: NSObject, URLSessionDataDelegate {
        var onData: ((Data) -> Void)?
        func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                        didReceive data: Data) {
            onData?(data)
        }
    }
}

struct MJPEGView: UIViewRepresentable {
    let url: URL?

    func makeUIView(context: Context) -> MJPEGStreamView {
        let view = MJPEGStreamView()
        if let url { view.start(url: url) }
        return view
    }

    func updateUIView(_ view: MJPEGStreamView, context: Context) {
        if let url { view.start(url: url) } else { view.stop() }
    }

    static func dismantleUIView(_ view: MJPEGStreamView, coordinator: ()) {
        view.stop()
    }
}
