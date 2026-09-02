import Darwin
import Foundation

/// Fire-and-forget UDP to the edge (:9000). IPv4 only; no receive path.
final class UDPClient: @unchecked Sendable {
    private let lock = NSLock()
    private var fd: Int32 = -1
    private var dest = sockaddr_in()

    @discardableResult
    func connect(host: String, port: UInt16) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if fd >= 0 {
            Darwin.close(fd)
            fd = -1
        }

        let trimmed = host.trimmingCharacters(in: .whitespacesAndNewlines)
        let ip = trimmed.split(separator: ":").first.map(String.init) ?? trimmed
        var addr = sockaddr_in()
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        let ok = ip.withCString { inet_pton(AF_INET, $0, &addr.sin_addr) }
        guard ok == 1 else { return false }

        let sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)
        guard sock >= 0 else { return false }
        fd = sock
        dest = addr
        return true
    }

    func send(_ data: Data) {
        lock.lock()
        let sock = fd
        var addr = dest
        lock.unlock()
        guard sock >= 0 else { return }
        data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            _ = withUnsafePointer(to: &addr) { ptr in
                ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                    sendto(
                        sock,
                        base,
                        raw.count,
                        0,
                        sa,
                        socklen_t(MemoryLayout<sockaddr_in>.size)
                    )
                }
            }
        }
    }

    func close() {
        lock.lock()
        if fd >= 0 {
            Darwin.close(fd)
            fd = -1
        }
        lock.unlock()
    }
}
