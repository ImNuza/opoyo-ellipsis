import Foundation

/// Binary PCM datagram for edge UDP port+1. Not a WAV file.
enum PCMFrame {
    static let magic = Data([0x4F, 0x50, 0x59, 0x41]) // OPYA
    static let version: UInt8 = 1
    static let rate: UInt16 = 16_000

    static func pack(nodeId: String, seq: UInt32, tMs: UInt64, pcm: Data) -> Data {
        var data = Data()
        data.reserveCapacity(37 + pcm.count)
        data.append(magic)
        data.append(version)
        var seqLE = seq.littleEndian
        data.append(Data(bytes: &seqLE, count: 4))
        var tLE = tMs.littleEndian
        data.append(Data(bytes: &tLE, count: 8))
        data.append(uuidBytes(nodeId))
        var rateLE = rate.littleEndian
        data.append(Data(bytes: &rateLE, count: 2))
        var nLE = UInt16(pcm.count / 2).littleEndian
        data.append(Data(bytes: &nLE, count: 2))
        data.append(pcm)
        return data
    }

    static func uuidBytes(_ value: String) -> Data {
        guard let parsed = UUID(uuidString: value) else {
            return Data(count: 16)
        }
        let u = parsed.uuid
        return Data([
            u.0, u.1, u.2, u.3, u.4, u.5, u.6, u.7,
            u.8, u.9, u.10, u.11, u.12, u.13, u.14, u.15,
        ])
    }
}
