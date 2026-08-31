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
    let soga: Color
    let line: Color
    let kawung: Color
    let onAccent: Color

    static let day = Cloth(
        paper: Color(hex: 0xF3E6C9),
        resist: Color(hex: 0xFBF6EA),
        ink: Color(hex: 0x2A1C12),
        muted: Color(hex: 0x6B5344),
        indigo: Color(hex: 0x1E2F4F),
        soga: Color(hex: 0x8B4B28),
        line: Color(hex: 0xCDB892),
        kawung: Color(hex: 0x1E2F4F).opacity(0.28),
        onAccent: Color(hex: 0xFBF6EA)
    )

    static let night = Cloth(
        paper: Color(hex: 0x1A140F),
        resist: Color(hex: 0x241C16),
        ink: Color(hex: 0xF3E6C9),
        muted: Color(hex: 0xC4B48A),
        indigo: Color(hex: 0xC4B48A),
        soga: Color(hex: 0xC47A4A),
        line: Color(hex: 0x3D3228),
        kawung: Color(hex: 0xC4B48A).opacity(0.22),
        onAccent: Color(hex: 0x1A140F)
    )

    static func make(_ scheme: ColorScheme) -> Cloth {
        scheme == .dark ? .night : .day
    }
}

enum Typeface {
    static func display(_ size: CGFloat) -> Font {
        .custom("IowanOldStyle-Bold", size: size)
    }

    static func displayRoman(_ size: CGFloat) -> Font {
        .custom("IowanOldStyle-Roman", size: size)
    }

    static func body(_ size: CGFloat) -> Font {
        .custom("AvenirNext-Regular", size: size)
    }

    static func bodyMedium(_ size: CGFloat) -> Font {
        .custom("AvenirNext-Medium", size: size)
    }

    static func bodyDemi(_ size: CGFloat) -> Font {
        .custom("AvenirNext-DemiBold", size: size)
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
