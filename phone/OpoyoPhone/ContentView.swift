import SwiftUI
import UIKit

struct ContentView: View {
    @Environment(\.colorScheme) private var colorScheme
    @State private var session = SensorSession()
    @State private var portText = ""
    @State private var keepAwake = false

    private var cloth: Cloth { .make(colorScheme) }
    private var hostIsEmpty: Bool {
        session.host.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        ZStack {
            cloth.paper.ignoresSafeArea()

            VStack(spacing: 0) {
                headerBar
                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        collect
                        liveReadout
                        axes
                        destination
                        controls
                        footnote
                    }
                    .padding(.horizontal, 20)
                    .padding(.top, 16)
                    .padding(.bottom, 40)
                }
                .scrollDismissesKeyboard(.interactively)
            }
        }
        .tint(cloth.indigo)
        .onAppear {
            portText = String(session.port)
            applyIdleTimer()
        }
        .onChange(of: session.isRunning) { _, _ in
            applyIdleTimer()
        }
        .onChange(of: session.isCapturing) { _, _ in
            applyIdleTimer()
        }
        .onChange(of: keepAwake) { _, _ in
            applyIdleTimer()
        }
        .onDisappear {
            if !session.isRunning && !keepAwake {
                UIApplication.shared.isIdleTimerDisabled = false
            }
        }
    }

    private var headerBar: some View {
        ZStack {
            Image("BatikHeader")
                .resizable()
                .scaledToFill()
                .overlay(cloth.header.opacity(0.72))
            HStack(alignment: .center, spacing: 10) {
                LogoMark()
                    .frame(width: 36, height: 36)
                VStack(alignment: .leading, spacing: 1) {
                    Text("OPOYO")
                        .font(Typeface.display(18))
                        .tracking(2.4)
                        .foregroundStyle(cloth.onHeader)
                    Text("Observe · Predict · On Your Own")
                        .font(Typeface.body(10))
                        .foregroundStyle(cloth.tan)
                        .lineLimit(1)
                        .minimumScaleFactor(0.6)
                }
                Spacer(minLength: 8)
                statusChip
            }
            .padding(.horizontal, 16)
        }
        .frame(maxWidth: .infinity)
        .frame(height: 56)
        .clipped()
    }

    private var statusChip: some View {
        let live = session.isRunning || session.isCapturing
        return HStack(spacing: 8) {
            Circle()
                .fill(live ? cloth.tan : cloth.onHeader.opacity(0.45))
                .frame(width: 8, height: 8)
            Text(session.isCapturing ? "Recording" : (session.isRunning ? "Streaming" : "Idle"))
                .font(Typeface.bodyMedium(12))
                .tracking(1.1)
                .textCase(.uppercase)
                .foregroundStyle(live ? cloth.tan : cloth.onHeader.opacity(0.8))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .overlay(
            RoundedRectangle(cornerRadius: 2, style: .continuous)
                .stroke(live ? cloth.tan : cloth.onHeader.opacity(0.35), lineWidth: 1)
        )
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(
            session.isCapturing ? "Recording" : (session.isRunning ? "Streaming" : "Idle")
        )
    }

    private var collect: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Collect a take")
                .font(Typeface.bodyMedium(11))
                .tracking(1.3)
                .textCase(.uppercase)
                .foregroundStyle(cloth.muted)
            Text("2 s still · one action · 2 s still · stop · send")
                .font(Typeface.body(13))
                .foregroundStyle(cloth.muted)

            Picker("Label", selection: $session.label) {
                ForEach(SensorSession.labels, id: \.self) { name in
                    Text(name).tag(name)
                }
            }
            .pickerStyle(.menu)
            .disabled(session.isCapturing)

            Button(action: toggleTake) {
                Text(session.isCapturing ? "Stop take · \(session.takeRows)" : "Record \(session.label)")
                    .font(Typeface.bodyDemi(18))
                    .frame(maxWidth: .infinity, minHeight: 56)
                    .foregroundStyle(cloth.onAccent)
                    .background(session.isCapturing ? cloth.soga : cloth.indigo)
            }
            .buttonStyle(.plain)

            if let url = session.lastTakeURL {
                ShareLink(item: url, preview: SharePreview(url.lastPathComponent)) {
                    Text("Send \(url.lastPathComponent)")
                        .font(Typeface.bodyDemi(16))
                        .frame(maxWidth: .infinity, minHeight: 48)
                        .foregroundStyle(cloth.indigo)
                        .background(cloth.resist)
                        .overlay(
                            RoundedRectangle(cornerRadius: 2, style: .continuous)
                                .stroke(cloth.indigo, lineWidth: 1)
                        )
                }
            }
        }
    }

    private var liveReadout: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("\(session.modelName) · \(session.shortId)")
                .font(Typeface.body(13))
                .foregroundStyle(cloth.muted)
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
    }

    private func liveTile(label: String, value: String, unit: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(label)
                .font(Typeface.bodyMedium(11))
                .tracking(1.3)
                .textCase(.uppercase)
                .foregroundStyle(cloth.muted)
            Text(value)
                .font(Typeface.number(40))
                .foregroundStyle(cloth.ink)
                .lineLimit(1)
                .minimumScaleFactor(0.45)
                .frame(maxWidth: .infinity, alignment: .leading)
            Text(unit)
                .font(Typeface.bodyMedium(13))
                .foregroundStyle(cloth.muted)
        }
        .padding(14)
        .frame(maxWidth: .infinity, minHeight: 112, alignment: .topLeading)
        .background(cloth.resist)
        .overlay(
            RoundedRectangle(cornerRadius: 2, style: .continuous)
                .stroke(cloth.line, lineWidth: 1)
        )
        .accessibilityElement(children: .combine)
    }

    private var axes: some View {
        HStack(spacing: 0) {
            axisCell("ax", String(format: "%+.3f", session.ax), "g")
            axisDivider
            axisCell("ay", String(format: "%+.3f", session.ay), "g")
            axisDivider
            axisCell("az", String(format: "%+.3f", session.az), "g")
            axisDivider
            axisCell("Packets", "\(session.packetsSent)", "")
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

    private func axisCell(_ label: String, _ value: String, _ unit: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(Typeface.bodyMedium(10))
                .tracking(1.1)
                .textCase(.uppercase)
                .foregroundStyle(cloth.muted)
            HStack(alignment: .firstTextBaseline, spacing: 4) {
                Text(value)
                    .font(Typeface.numberRegular(16))
                    .foregroundStyle(cloth.ink)
                    .lineLimit(1)
                    .minimumScaleFactor(0.6)
                if !unit.isEmpty {
                    Text(unit)
                        .font(Typeface.bodyMedium(11))
                        .foregroundStyle(cloth.muted)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 10)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(label) \(value) \(unit)")
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

            Text("JSON \(session.port) · PCM \(Int(session.port) + 1)")
                .font(Typeface.bodyMedium(11))
                .foregroundStyle(cloth.muted)
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

            Button(action: {
                keepAwake.toggle()
                applyIdleTimer()
            }) {
                Text(keepAwake ? "Screen stays on" : "Keep screen awake")
                    .font(Typeface.bodyDemi(16))
                    .frame(maxWidth: .infinity, minHeight: 48)
                    .foregroundStyle(cloth.indigo)
                    .background(cloth.resist)
                    .overlay(
                        RoundedRectangle(cornerRadius: 2, style: .continuous)
                            .stroke(keepAwake || session.isRunning ? cloth.indigo : cloth.line, lineWidth: 1)
                    )
            }
            .buttonStyle(.plain)
            .accessibilityHint("Keeps the screen on after Stop. Streaming already keeps it on.")

            Text(session.status)
                .font(Typeface.body(14))
                .foregroundStyle(session.isRunning ? cloth.indigo : cloth.muted)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var footnote: some View {
        Text("Collect: face-down on tile, pick a label, Record, one action, Stop, Send on Telegram. Laptop streaming is only for the live dashboard.")
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
        applyIdleTimer()
    }

    private func toggleTake() {
        if session.isCapturing {
            _ = session.stopTake()
        } else {
            session.startTake()
        }
        applyIdleTimer()
    }

    private func applyIdleTimer() {
        UIApplication.shared.isIdleTimerDisabled = session.isRunning || session.isCapturing || keepAwake
    }
}
