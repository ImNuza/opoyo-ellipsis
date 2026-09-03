import Darwin
import Foundation
import UIKit

/// Stable phone UUID in UserDefaults. This is SensorSample.id / the edge hub key.
enum DeviceIdentity {
    static var nodeId: String {
        let key = "opoyo.nodeId"
        if let existing = UserDefaults.standard.string(forKey: key), !existing.isEmpty {
            return existing
        }
        let fresh = UUID().uuidString.lowercased()
        UserDefaults.standard.set(fresh, forKey: key)
        return fresh
    }

    static var shortId: String {
        String(nodeId.replacingOccurrences(of: "-", with: "").prefix(4))
    }

    static var modelName: String {
        var sys = utsname()
        uname(&sys)
        var machine = sys.machine
        let identifier = withUnsafeBytes(of: &machine) { raw -> String in
            let ptr = raw.bindMemory(to: CChar.self)
            guard let base = ptr.baseAddress else { return "" }
            return String(cString: base)
        }
        return Self.map[identifier] ?? (identifier.isEmpty ? UIDevice.current.model : identifier)
    }

    private static let map: [String: String] = [
        "iPhone17,1": "iPhone 16 Pro",
        "iPhone17,2": "iPhone 16 Pro Max",
        "iPhone17,3": "iPhone 16",
        "iPhone17,4": "iPhone 16 Plus",
        "iPhone18,1": "iPhone 17 Pro",
        "iPhone18,2": "iPhone 17 Pro Max",
        "iPhone18,3": "iPhone 17",
        "iPhone16,1": "iPhone 15 Pro",
        "iPhone16,2": "iPhone 15 Pro Max",
        "iPhone15,4": "iPhone 15",
        "iPhone15,5": "iPhone 15 Plus",
        "iPhone15,2": "iPhone 14 Pro",
        "iPhone15,3": "iPhone 14 Pro Max",
        "iPhone14,7": "iPhone 14",
        "iPhone14,8": "iPhone 14 Plus",
        "i386": "Simulator",
        "x86_64": "Simulator",
        "arm64": "Simulator",
    ]
}
