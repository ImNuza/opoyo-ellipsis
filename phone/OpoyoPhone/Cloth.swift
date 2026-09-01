import SwiftUI

extension Color {
    init(hex: UInt32) {
        let r = Double((hex >> 16) & 0xFF) / 255
        let g = Double((hex >> 8) & 0xFF) / 255
        let b = Double(hex & 0xFF) / 255
        self.init(.sRGB, red: r, green: g, blue: b, opacity: 1)
    }
}

struct Cloth {
    let paper: Color
    let resist: Color
    let ink: Color
    let muted: Color
    let indigo: Color
    let teal: Color
    let tan: Color
    let soga: Color
    let line: Color
    let kawung: Color
    let onAccent: Color
    let header: Color
    let onHeader: Color

    static let day = Cloth(
        paper: Color(hex: 0xFAF7F1),
        resist: Color(hex: 0xFFFFFF),
        ink: Color(hex: 0x1E3A5F),
        muted: Color(hex: 0x5C6B7A),
        indigo: Color(hex: 0x1E3A5F),
        teal: Color(hex: 0x3D7A7A),
        tan: Color(hex: 0xC4A574),
        soga: Color(hex: 0xC4A574),
        line: Color(hex: 0xD8D0C4),
        kawung: Color(hex: 0xC4A574).opacity(0.35),
        onAccent: Color(hex: 0xFAF7F1),
        header: Color(hex: 0x1E3A5F),
        onHeader: Color(hex: 0xFAF7F1)
    )

    static let night = Cloth(
        paper: Color(hex: 0x121820),
        resist: Color(hex: 0x1A222C),
        ink: Color(hex: 0xFAF7F1),
        muted: Color(hex: 0xB7C0C8),
        indigo: Color(hex: 0xC4A574),
        teal: Color(hex: 0x7AA8A8),
        tan: Color(hex: 0xC4A574),
        soga: Color(hex: 0xC4A574),
        line: Color(hex: 0x314050),
        kawung: Color(hex: 0xC4A574).opacity(0.28),
        onAccent: Color(hex: 0x121820),
        header: Color(hex: 0x0E141C),
        onHeader: Color(hex: 0xFAF7F1)
    )

    static func make(_ scheme: ColorScheme) -> Cloth {
        scheme == .dark ? .night : .day
    }
}

enum Typeface {
    static func display(_ size: CGFloat) -> Font {
        .system(size: size, weight: .bold, design: .default)
    }

    static func displayRoman(_ size: CGFloat) -> Font {
        .system(size: size, weight: .regular, design: .default)
    }

    static func body(_ size: CGFloat) -> Font {
        .system(size: size, weight: .regular, design: .default)
    }

    static func bodyMedium(_ size: CGFloat) -> Font {
        .system(size: size, weight: .medium, design: .default)
    }

    static func bodyDemi(_ size: CGFloat) -> Font {
        .system(size: size, weight: .semibold, design: .default)
    }

    static func number(_ size: CGFloat) -> Font {
        .system(size: size, weight: .semibold, design: .rounded).monospacedDigit()
    }

    static func numberRegular(_ size: CGFloat) -> Font {
        .system(size: size, weight: .medium, design: .rounded).monospacedDigit()
    }
}

struct KawungCloth: View {
    var stroke: Color
    var tile: CGFloat = 72

    var body: some View {
        Canvas { context, size in
            let cols = Int(ceil(size.width / tile)) + 1
            let rows = Int(ceil(size.height / tile)) + 1
            let scale = tile / 72
            let rBig = 26 * scale
            let rSmall = 7.5 * scale
            let lineWidth = 1.15 * scale

            for row in 0..<rows {
                for col in 0..<cols {
                    let origin = CGPoint(x: CGFloat(col) * tile, y: CGFloat(row) * tile)
                    var path = Path()
                    let centers = [
                        CGPoint(x: origin.x, y: origin.y + tile / 2),
                        CGPoint(x: origin.x + tile, y: origin.y + tile / 2),
                        CGPoint(x: origin.x + tile / 2, y: origin.y),
                        CGPoint(x: origin.x + tile / 2, y: origin.y + tile)
                    ]
                    for center in centers {
                        path.addEllipse(in: CGRect(
                            x: center.x - rBig,
                            y: center.y - rBig,
                            width: rBig * 2,
                            height: rBig * 2
                        ))
                    }
                    let mid = CGPoint(x: origin.x + tile / 2, y: origin.y + tile / 2)
                    path.addEllipse(in: CGRect(
                        x: mid.x - rSmall,
                        y: mid.y - rSmall,
                        width: rSmall * 2,
                        height: rSmall * 2
                    ))
                    context.stroke(path, with: .color(stroke), lineWidth: lineWidth)
                }
            }
        }
        .accessibilityHidden(true)
        .allowsHitTesting(false)
    }
}

struct LogoMark: View {
    var body: some View {
        Canvas { context, size in
            let w = size.width
            let h = size.height
            func lobe(_ rect: CGRect, color: Color) {
                let p = Path(ellipseIn: rect)
                context.fill(p, with: .color(color))
            }
            lobe(CGRect(x: w * 0.08, y: h * 0.18, width: w * 0.84, height: h * 0.62), color: Color(hex: 0xC4A574))
            lobe(CGRect(x: w * 0.18, y: h * 0.26, width: w * 0.64, height: h * 0.48), color: Color(hex: 0x3D7A7A))
            lobe(CGRect(x: w * 0.28, y: h * 0.34, width: w * 0.44, height: h * 0.36), color: Color(hex: 0x1E3A5F))
            let sun = CGRect(x: w * 0.42, y: h * 0.44, width: w * 0.16, height: w * 0.16)
            context.fill(Path(ellipseIn: sun), with: .color(Color(hex: 0xFAF7F1)))
            var base = Path()
            base.move(to: CGPoint(x: w * 0.12, y: h * 0.9))
            base.addLine(to: CGPoint(x: w * 0.88, y: h * 0.9))
            context.stroke(base, with: .color(Color(hex: 0x1E3A5F)), lineWidth: 1.4)
        }
        .accessibilityHidden(true)
    }
}
