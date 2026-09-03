# OPOYO Android collector

Same job as the iOS Collect screen: record one take on the phone, share the CSV (Telegram is fine).

Open `src/phone/android/` in Android Studio → Run on a physical phone. Allow microphone.

CSV header: `t,ax,ay,az,mag,db`. Axes are in **g** (Android linear acceleration ÷ 9.81), same as iOS.

Dump received files into `data/` on the laptop.
