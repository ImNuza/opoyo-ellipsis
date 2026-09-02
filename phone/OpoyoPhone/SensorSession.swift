import AVFoundation
import CoreMotion
import Foundation
import Observation

@Observable
final class SensorSession {
    var host: String
    var port: UInt16
    var isRunning = false
    var magnitude: Double = 0
    var ax: Double = 0
    var ay: Double = 0
    var az: Double = 0
    var decibels: Double = -120
    var packetsSent = 0
    var status: String = "Idle"
    var isCapturing = false
    var takeRows = 0
    var lastTakeURL: URL?
    var label = "heeldrop"
    let nodeId = DeviceIdentity.nodeId
    let shortId = DeviceIdentity.shortId
    let modelName = DeviceIdentity.modelName

    static let labels = ["heeldrop", "bag", "book", "pan", "door", "walk", "quiet", "tv"]

    private let motion = CMMotionManager()
    private let engine = AVAudioEngine()
    private let client = UDPClient()
    private let meter = DbMeter()
    private var tapInstalled = false
    private var streamUDP = false
    private var sensingForTakeOnly = false
    private var capture: [(Int, Double, Double, Double, Double, Double)] = []

    init() {
        let stored = UserDefaults.standard.string(forKey: "opoyo.host") ?? ""
        host = stored.isEmpty ? "172.20.10.11" : stored
        let storedPort = UserDefaults.standard.integer(forKey: "opoyo.port")
        port = storedPort > 0 ? UInt16(storedPort) : 9000
    }

    func start(stream: Bool = true) {
        streamUDP = stream
        UserDefaults.standard.set(host, forKey: "opoyo.host")
        UserDefaults.standard.set(Int(port), forKey: "opoyo.port")

        Task { [weak self] in
            guard let self else { return }
            let granted = await AVAudioApplication.requestRecordPermission()
            if !granted {
                self.status = "Mic permission denied"
                return
            }
            self.beginSensing()
        }
    }

    func startTake() {
        capture.removeAll()
        takeRows = 0
        lastTakeURL = nil
        isCapturing = true
        if isRunning {
            status = "Recording \(label)…"
            return
        }
        sensingForTakeOnly = true
        start(stream: false)
    }

    func stopTake() -> URL? {
        isCapturing = false
        let url = writeCSV()
        lastTakeURL = url
        takeRows = capture.count
        if sensingForTakeOnly {
            sensingForTakeOnly = false
            stop()
        }
        if let url {
            status = "Saved \(url.lastPathComponent). Send it."
        } else {
            status = "No samples — try again"
        }
        return url
    }

    func stop() {
        motion.stopDeviceMotionUpdates()
        if tapInstalled {
            engine.inputNode.removeTap(onBus: 0)
            tapInstalled = false
        }
        if engine.isRunning {
            engine.stop()
        }
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        client.close()
        isRunning = false
        if !isCapturing {
            status = "Stopped"
        }
    }

    private func beginSensing() {
        do {
            try configureAudio()
        } catch {
            status = "Audio failed: \(error.localizedDescription)"
            return
        }

        guard motion.isDeviceMotionAvailable else {
            status = "Motion unavailable"
            return
        }

        if streamUDP {
            guard client.connect(host: host, port: port) else {
                status = "Bad Mac address. Use dotted IPv4, not a name."
                return
            }
        }
        motion.deviceMotionUpdateInterval = 1.0 / 50.0
        motion.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: .main) { [weak self] data, error in
            guard let self else { return }
            if let error {
                self.status = error.localizedDescription
                return
            }
            guard let data else { return }
            self.emit(data)
        }

        isRunning = true
        if isCapturing {
            status = "Recording \(label)…"
        } else if streamUDP {
            status = "Streaming to \(host):\(port)"
        } else {
            status = "Sensing"
        }
    }

    private func configureAudio() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .measurement, options: [.mixWithOthers, .defaultToSpeaker])
        try session.setActive(true)

        if tapInstalled {
            engine.inputNode.removeTap(onBus: 0)
            tapInstalled = false
        }

        let input = engine.inputNode
        let format = input.inputFormat(forBus: 0)
        let dbMeter = meter
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in
            guard let channel = buffer.floatChannelData?[0] else { return }
            let n = Int(buffer.frameLength)
            guard n > 0 else { return }
            var sum: Float = 0
            for i in 0..<n {
                let sample = channel[i]
                sum += sample * sample
            }
            let rms = sqrt(sum / Float(n))
            let db = Double(20 * log10(max(rms, 1e-8)))
            dbMeter.set(db)
        }
        tapInstalled = true
        try engine.start()
    }

    private func emit(_ data: CMDeviceMotion) {
        let ua = data.userAcceleration
        let mag = sqrt(ua.x * ua.x + ua.y * ua.y + ua.z * ua.z)
        let db = meter.get()
        ax = ua.x
        ay = ua.y
        az = ua.z
        magnitude = mag
        decibels = db

        let t = Int(Date().timeIntervalSince1970 * 1000)
        if isCapturing {
            capture.append((t, ua.x, ua.y, ua.z, mag, db))
            takeRows = capture.count
        }
        if streamUDP {
            let payload =
                "{\"v\":2,\"id\":\"\(esc(nodeId))\",\"model\":\"\(esc(modelName))\",\"t\":\(t),\"ax\":\(fmt(ua.x)),\"ay\":\(fmt(ua.y)),\"az\":\(fmt(ua.z)),\"mag\":\(fmt(mag)),\"db\":\(fmt(db))}"
            if let bytes = payload.data(using: .utf8) {
                client.send(bytes)
                packetsSent += 1
            }
        }
    }

    private func writeCSV() -> URL? {
        guard !capture.isEmpty else { return nil }
        let fmtDate = DateFormatter()
        fmtDate.locale = Locale(identifier: "en_US_POSIX")
        fmtDate.dateFormat = "yyyyMMdd_HHmmss"
        let stamp = fmtDate.string(from: Date())
        let safe = label.replacingOccurrences(of: " ", with: "").lowercased()
        let name = "\(safe)_\(stamp).csv"
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("takes", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent(name)
        var body = "t,ax,ay,az,mag,db\n"
        for row in capture {
            body += "\(row.0),\(fmt(row.1)),\(fmt(row.2)),\(fmt(row.3)),\(fmt(row.4)),\(fmt(row.5))\n"
        }
        do {
            try body.write(to: url, atomically: true, encoding: .utf8)
            return url
        } catch {
            status = "Write failed: \(error.localizedDescription)"
            return nil
        }
    }

    private func fmt(_ value: Double) -> String {
        String(format: "%.4f", value)
    }

    private func esc(_ value: String) -> String {
        value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
    }
}
