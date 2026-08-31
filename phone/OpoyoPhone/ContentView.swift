import SwiftUI
import UIKit

struct ContentView: View {
    @Environment(\.colorScheme) private var colorScheme
    @State private var session = SensorSession()
    @State private var portText = ""

    private var cloth: Cloth { .make(colorScheme) }
    private var hostIsEmpty: Bool {
        session.host.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        ZStack {
            cloth.paper.ignoresSafeArea()
            KawungCloth(stroke: cloth.kawung)
                .ignoresSafeArea()

            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    header
                    liveReadout
                    axes
                    destination
                    controls
                    footnote
                }
                .padding(.horizontal, 20)
                .padding(.top, 16)
                .padding(.bottom, 32)
            }
            .scrollDismissesKeyboard(.interactively)
        }
        .tint(cloth.indigo)
        .onAppear {
            portText = String(session.port)
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                Text("OPOYO")
                    .font(Typeface.display(34))
                    .foregroundStyle(cloth.ink)
                    .tracking(-0.4)
                Spacer(minLength: 12)
                statusChip
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Phone is a proxy.")
                    .font(Typeface.displayRoman(17))
                    .foregroundStyle(cloth.ink)
                Text("Product is a floor puck.")
                    .font(Typeface.body(15))
                    .foregroundStyle(cloth.muted)
            }
        }
    }

    private var statusChip: some View {
        let streaming = session.isRunning
        return HStack(spacing: 8) {
            Circle()
                .fill(streaming ? cloth.indigo : cloth.muted.opacity(0.45))
                .frame(width: 8, height: 8)
            Text(streaming ? "Streaming" : "Idle")
                .font(Typeface.bodyMedium(12))
                .tracking(1.1)
                .textCase(.uppercase)
                .foregroundStyle(streaming ? cloth.indigo : cloth.muted)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(cloth.resist)
        .overlay(
            RoundedRectangle(cornerRadius: 2, style: .continuous)
                .stroke(streaming ? cloth.indigo : cloth.line, lineWidth: 1)
        )
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(streaming ? "Streaming" : "Idle")
    }

    private var liveReadout: some View {
        HStack(alignment: .top, spacing: 12) {
            liveTile(
                label: "Magnitude",
                value: String(format: "%.3f", session.magnitude),
                unit: "g"
            )
            liveTile(
                label: "Sound",
                value: String(format: "%.1f", session.decibels),
                unit: "dB"
            )
        }
    }

    private func liveTile(label: String, value: String, unit: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(label)
                .font(Typeface.bodyMedium(11))
                .tracking(1.3)
                .textCase(.uppercase)
                .foregroundStyle(cloth.muted)
            Text(value)
                .font(Typeface.number(56))
                .foregroundStyle(cloth.ink)
                .lineLimit(1)
                .minimumScaleFactor(0.45)
                .frame(maxWidth: .infinity, alignment: .leading)
            Text(unit)
                .font(Typeface.bodyMedium(13))
                .foregroundStyle(cloth.muted)
        }
        .padding(16)
        .frame(maxWidth: .infinity, minHeight: 148, alignment: .topLeading)
        .background(cloth.resist)
        .overlay(
            RoundedRectangle(cornerRadius: 2, style: .continuous)
                .stroke(cloth.line, lineWidth: 1)
        )
        .accessibilityElement(children: .combine)
    }

    private var axes: some View {
        HStack(spacing: 0) {
            axisCell("ax", String(format: "%+.3f", session.ax))
            axisDivider
            axisCell("ay", String(format: "%+.3f", session.ay))
            axisDivider
            axisCell("az", String(format: "%+.3f", session.az))
            axisDivider
            axisCell("Packets", "\(session.packetsSent)")
        }
        .padding(.vertical, 12)
        .padding(.horizontal, 4)
        .background(cloth.resist.opacity(0.7))
        .overlay(
            RoundedRectangle(cornerRadius: 2, style: .continuous)
                .stroke(cloth.line, lineWidth: 1)
        )
    }

    private var axisDivider: some View {
        Rectangle()
            .fill(cloth.line)
            .frame(width: 1, height: 36)
    }

    private func axisCell(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(Typeface.bodyMedium(10))
                .tracking(1.1)
                .textCase(.uppercase)
                .foregroundStyle(cloth.muted)
            Text(value)
                .font(Typeface.numberRegular(16))
                .foregroundStyle(cloth.ink)
                .lineLimit(1)
                .minimumScaleFactor(0.6)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 10)
        .accessibilityElement(children: .combine)
    }

    private var destination: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Mac address")
                .font(Typeface.bodyMedium(11))
                .tracking(1.3)
                .textCase(.uppercase)
                .foregroundStyle(cloth.muted)

            clothField(
                placeholder: "10.0.0.1",
                text: $session.host,
                keyboard: .numbersAndPunctuation,
                disableAutocap: true
            )
            .disabled(session.isRunning)

            clothField(
                placeholder: "9000",
                text: $portText,
                keyboard: .numberPad,
                disableAutocap: false
            )
            .disabled(session.isRunning)
            .onChange(of: portText) { _, newValue in
                if let parsed = UInt16(newValue), parsed > 0 {
                    session.port = parsed
                }
            }
        }
    }

    private func clothField(
        placeholder: String,
        text: Binding<String>,
        keyboard: UIKeyboardType,
        disableAutocap: Bool
    ) -> some View {
        TextField(placeholder, text: text)
            .font(Typeface.numberRegular(18))
            .foregroundStyle(cloth.ink)
            .keyboardType(keyboard)
            .textInputAutocapitalization(disableAutocap ? .never : .sentences)
            .autocorrectionDisabled(disableAutocap)
            .padding(.horizontal, 12)
            .frame(minHeight: 48)
            .background(cloth.resist)
            .overlay(
                RoundedRectangle(cornerRadius: 2, style: .continuous)
                    .stroke(cloth.line, lineWidth: 1)
            )
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 10) {
            Button(action: toggle) {
                Text(session.isRunning ? "Stop" : "Start")
                    .font(Typeface.bodyDemi(18))
                    .frame(maxWidth: .infinity, minHeight: 56)
                    .foregroundStyle(cloth.onAccent)
                    .background(session.isRunning ? cloth.soga : cloth.indigo)
            }
            .buttonStyle(.plain)
            .disabled(hostIsEmpty)
            .opacity(hostIsEmpty ? 0.4 : 1)
            .accessibilityHint(
                session.isRunning
                    ? "Stops accelerometer and microphone streaming"
                    : "Starts accelerometer and microphone streaming"
            )

            Text(session.status)
                .font(Typeface.body(14))
                .foregroundStyle(session.isRunning ? cloth.indigo : cloth.muted)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var footnote: some View {
        Text("Place the phone face-down on tile. Same Wi-Fi or personal hotspot as the Mac. Allow microphone and local network when iOS asks.")
            .font(Typeface.body(13))
            .foregroundStyle(cloth.muted)
            .fixedSize(horizontal: false, vertical: true)
    }

    private func toggle() {
        if session.isRunning {
            session.stop()
        } else {
            session.start()
        }
    }
}
