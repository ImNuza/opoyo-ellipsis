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
    let nodeId = DeviceIdentity.nodeId
    let shortId = DeviceIdentity.shortId
    let modelName = DeviceIdentity.modelName

    private let motion = CMMotionManager()
    private let engine = AVAudioEngine()
    private let client = UDPClient()
    private let meter = DbMeter()
    private var tapInstalled = false

    init() {
        let stored = UserDefaults.standard.string(forKey: "opoyo.host") ?? ""
        host = stored.isEmpty ? "172.20.10.11" : stored
        let storedPort = UserDefaults.standard.integer(forKey: "opoyo.port")
        port = storedPort > 0 ? UInt16(storedPort) : 9000
    }

    func start() {
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
        status = "Stopped"
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

        client.connect(host: host, port: port)
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
        status = "Streaming to \(host):\(port)"
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
        let payload =
            "{\"v\":2,\"id\":\"\(esc(nodeId))\",\"model\":\"\(esc(modelName))\",\"t\":\(t),\"ax\":\(fmt(ua.x)),\"ay\":\(fmt(ua.y)),\"az\":\(fmt(ua.z)),\"mag\":\(fmt(mag)),\"db\":\(fmt(db))}"
        if let bytes = payload.data(using: .utf8) {
            client.send(bytes)
            packetsSent += 1
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
