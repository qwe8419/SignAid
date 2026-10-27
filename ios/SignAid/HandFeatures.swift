import Foundation
import Vision

/// Swift port of features.py. The layout must match the training code exactly
/// -- the model has no way to tell a reordered feature vector from a different
/// hand shape, it just produces confident nonsense.
///
/// 300 features: 63 canonicalised coordinates, 210 pairwise distances,
/// 15 joint angles, 5 fingertip angles, 5 fingertip heights, and 2 ORIENTATION
/// features (sin/cos of the raw hand angle before canonicalisation).
///
/// This must stay at 300 and in this order: the shipped weights are
/// letter_net_weights.npz, the exact model pi_standalone.py runs on the
/// glasses. Six crossing features were added here at one point for R; they are
/// gone again because the glasses model has no inputs for them.
///
/// The orientation pair matters because canonicalising rotation — which is what
/// makes everything else robust to a tilted camera — also discards the only
/// thing separating letters that share a handshape and differ in direction.
/// H and U are the same shape 49 degrees apart; so are G and Q.

enum HandFeatures {

    private static let wrist = 0
    private static let middleMCP = 9
    private static let chains = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
                                 [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
    private static let tips = [4, 8, 12, 16, 20]

    /// Vision's joint names in MediaPipe's index order. Vision reports named
    /// joints, MediaPipe reports an array, and they happen to describe the same
    /// 21 points -- but only in this order.
    private static let visionOrder: [VNHumanHandPoseObservation.JointName] = [
        .wrist,
        .thumbCMC, .thumbMP, .thumbIP, .thumbTip,
        .indexMCP, .indexPIP, .indexDIP, .indexTip,
        .middleMCP, .middlePIP, .middleDIP, .middleTip,
        .ringMCP, .ringPIP, .ringDIP, .ringTip,
        .littleMCP, .littlePIP, .littleDIP, .littleTip,
    ]

    /// -> 21 points as [x, y, z]. Vision has no depth, so z is 0 and the shipped
    /// model is trained the same way.
    static func orderedPoints(
        from points: [VNHumanHandPoseObservation.JointName: VNRecognizedPoint]
    ) -> [[Double]]? {
        var out: [[Double]] = []
        for name in visionOrder {
            guard let p = points[name], p.confidence > 0.3 else { return nil }
            // Vision's origin is bottom-left; MediaPipe's is top-left.
            out.append([Double(p.location.x), 1.0 - Double(p.location.y), 0.0])
        }
        return out
    }

    /// Phone landmarks -> the frame the model was trained in. Matches
    /// to_model_frame in train_robust.py; apply it to the (already mirrored)
    /// points before extract(), and to anything recorded for training.
    ///
    /// MediaPipe returns these in the camera SENSOR's frame: a landscape 640x480
    /// buffer, a quarter-turn from upright. That leaves the two orientation
    /// features 90 degrees off, which is why upright two-finger letters (U, R,
    /// K) all read as H. The x stretch moves the 4:3 squash back onto the axis
    /// the webcam training data had it on. On 800 recorded phone samples this
    /// took a webcam-only model from 0.25 to 0.99 per frame.
    static func toModelFrame(_ pts: [[Double]]) -> [[Double]] {
        let k = 16.0 / 9.0          // (4/3)^2
        return pts.map { [$0[1], -k * $0[0], k * $0[2]] }
    }

    /// Translate to the wrist, scale by hand size, rotate so wrist->middle-MCP
    /// points up. Removes position, scale and in-plane rotation.
    private static func canonicalize(_ pts: [[Double]]) -> [[Double]] {
        var p = pts.map { [$0[0] - pts[wrist][0], $0[1] - pts[wrist][1], $0[2] - pts[wrist][2]] }
        var scale = (p[middleMCP][0] * p[middleMCP][0]
                     + p[middleMCP][1] * p[middleMCP][1]).squareRoot()
        if scale < 1e-6 {
            scale = max(p.map { ($0[0] * $0[0] + $0[1] * $0[1]).squareRoot() }.max() ?? 1, 1e-6)
        }
        for i in p.indices { p[i] = [p[i][0] / scale, p[i][1] / scale, p[i][2] / scale] }

        let vx = p[middleMCP][0], vy = p[middleMCP][1]
        let n = (vx * vx + vy * vy).squareRoot()
        if n > 1e-6 {
            let cosA = -vy / n, sinA = -vx / n
            for i in p.indices {
                let x = p[i][0], y = p[i][1]
                p[i][0] = cosA * x - sinA * y
                p[i][1] = sinA * x + cosA * y
            }
        }
        return p
    }

    private static func angle(_ a: [Double], _ b: [Double], _ c: [Double]) -> Double {
        let v1 = [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
        let v2 = [c[0] - b[0], c[1] - b[1], c[2] - b[2]]
        let n1 = (v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]).squareRoot()
        let n2 = (v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]).squareRoot()
        guard n1 * n2 > 1e-9 else { return 0 }
        let dot = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (n1 * n2)
        return acos(max(-1, min(1, dot)))
    }

    static func extract(_ raw: [[Double]]) -> [Double] {
        // Measured BEFORE canonicalisation, which is the whole point.
        let v = [raw[middleMCP][0] - raw[wrist][0], raw[middleMCP][1] - raw[wrist][1]]
        let handAngle = atan2(v[1], v[0])

        let p = canonicalize(raw)
        var f: [Double] = []
        f.reserveCapacity(300)

        for pt in p { f.append(contentsOf: pt) }                 // 63

        for i in 0..<21 {                                        // 210
            for j in (i + 1)..<21 {
                let dx = p[i][0] - p[j][0], dy = p[i][1] - p[j][1], dz = p[i][2] - p[j][2]
                f.append((dx * dx + dy * dy + dz * dz).squareRoot())
            }
        }

        for chain in chains {                                    // 15
            for i in 0..<(chain.count - 2) {
                f.append(angle(p[chain[i]], p[chain[i + 1]], p[chain[i + 2]]))
            }
        }

        for t in tips { f.append(atan2(p[t][1], p[t][0])) }       // 5
        for t in tips { f.append(p[t][1]) }                       // 5
        f.append(sin(handAngle))                                  // 2
        f.append(cos(handAngle))
        return f
    }
}

// MARK: - Hand-shape coach measurements

extension HandFeatures {
    /// Port of coach_features.py, same order and units: per-finger bend,
    /// index-finger direction, index/middle spread and crossing, and three
    /// thumb placements. Input is in the model frame (toModelFrame applied).
    static func coachFeatures(_ raw: [[Double]]) -> [Double] {
        let p = canonicalize(raw)
        var out: [Double] = []
        for chain in chains {
            var bend = 0.0
            for i in 0..<(chain.count - 2) {
                bend += Double.pi - angle(p[chain[i]], p[chain[i + 1]], p[chain[i + 2]])
            }
            out.append(bend)
        }
        let vx = raw[8][0] - raw[5][0], vy = raw[8][1] - raw[5][1]
        let n = (vx * vx + vy * vy).squareRoot()
        out.append(n > 1e-9 ? vy / n : 0)

        let di = (p[8][0] - p[5][0], p[8][1] - p[5][1])
        let dm = (p[12][0] - p[9][0], p[12][1] - p[9][1])
        let den = max((di.0 * di.0 + di.1 * di.1).squareRoot() * (dm.0 * dm.0 + dm.1 * dm.1).squareRoot(), 1e-9)
        out.append(acos(max(-1, min(1, (di.0 * dm.0 + di.1 * dm.1) / den))))

        let side = p[5][0] - p[9][0]
        let sign: Double = side > 0 ? 1 : (side < 0 ? -1 : 0)
        out.append(sign * (p[8][0] - p[12][0]))

        func dist(_ a: [Double], _ b: [Double]) -> Double {
            ((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1])
             + (a[2] - b[2]) * (a[2] - b[2])).squareRoot()
        }
        out.append(dist(p[4], p[8]))
        out.append(dist(p[4], p[10]))
        out.append(abs(p[4][0]))
        return out
    }
}
