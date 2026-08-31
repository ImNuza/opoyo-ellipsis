import Foundation

final class DbMeter: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Double = -120

    func set(_ next: Double) {
        lock.lock()
        value = next
        lock.unlock()
    }

    func get() -> Double {
        lock.lock()
        defer { lock.unlock() }
        return value
    }
}
