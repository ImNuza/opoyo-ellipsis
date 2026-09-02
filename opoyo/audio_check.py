from __future__ import annotations

import numpy as np


class AudioCheck:
    """Second classifier: may suppress an alarm. Fail-open if disabled or broken.

    YAMNet is optional. Without a 16 kHz waveform (demo phone sends dB only),
    this uses a loudness-envelope heuristic or returns (False, '', 0).
    """

    def __init__(self, classes, threshold=0.30, enabled=True):
        self.classes = set(classes)
        self.threshold = threshold
        self.enabled = enabled
        self.ok = False
        self.model = None
        self.names: list[str] = []
        if enabled:
            self._try_load_yamnet()

    def _try_load_yamnet(self) -> None:
        try:
            import csv
            import tensorflow_hub as hub

            self.model = hub.load("https://tfhub.dev/google/yamnet/1")
            path = self.model.class_map_path().numpy().decode()
            with open(path, encoding="utf-8") as f:
                self.names = [r["display_name"] for r in csv.DictReader(f)]
            self.ok = True
        except Exception as e:
            print(f"[audio_check] YAMNet disabled: {e}")

    def check(self, audio) -> tuple[bool, str, float]:
        if not self.enabled:
            return False, "", 0.0
        try:
            w = np.asarray(audio, dtype=np.float32).ravel()
            if w.size == 0:
                return False, "", 0.0
            if self.ok and self.model is not None and w.size >= 1600:
                return self._yamnet(w)
            return self._envelope(w)
        except Exception as e:
            print(f"[audio_check] error, passing through: {e}")
            return False, "", 0.0

    def _yamnet(self, w: np.ndarray) -> tuple[bool, str, float]:
        if w.size < 16000:
            w = np.pad(w, (0, 16000 - w.size))
        w = np.clip(w, -1.0, 1.0)
        scores, _, _ = self.model(w)
        mean = scores.numpy().mean(axis=0)
        best_label, best_score = "", 0.0
        for i, s in enumerate(mean):
            if self.names[i] in self.classes and s > best_score:
                best_label, best_score = self.names[i], float(s)
        return (best_score >= self.threshold), best_label, best_score

    def _envelope(self, w: np.ndarray) -> tuple[bool, str, float]:
        """dB series: sustained loudness without an impulse is treated as TV-like."""
        if w.size < 8:
            return False, "", 0.0
        # Phone dB is typically negative (e.g. -40). Waveform PCM is ~[-1,1].
        if float(np.max(np.abs(w))) <= 1.5:
            return False, "", 0.0
        med = float(np.median(w))
        ptp = float(np.ptp(w))
        if med > -25.0 and ptp < 12.0:
            return True, "sustained_loudness", min(1.0, (med + 40.0) / 40.0)
        return False, "", 0.0
