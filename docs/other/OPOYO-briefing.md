# OPOYO briefing

Read this before a pitch. Every number is labelled: **physics** / **Alwan range** / **runbook heuristic** / **placeholder** / **unmeasured**. Do not cite our figure times as papers. Do not claim a range we have not measured.

## Problem

A senior falls, cannot get up, and cannot call. The harm is time on the floor (a long lie), not the impact itself. Wall buttons require reaching and pressing. Wearables are removed, uncharged, or refused, and are off while showering and sleeping. Cameras are rejected in the bedroom and bathroom. This segment is left with no working path to help.

## System

OPOYO is a sequential decision on floor vibration and air. It does not ask the senior to press, wear, or speak.

**Demo.** An iPhone on the floor streams **50 Hz** (runbook heuristic) user-acceleration in **g** (physics) plus a scalar sound level in **dB**. Packets are UDP JSON `{v,id,model,t,ax,ay,az,mag,db}` to the laptop on port **9000** (runbook heuristic). The laptop converts g to m/s² with **9.81** (physics).

**Production.** A piezo and a microphone run on the device. Only event JSON leaves the flat. Raw vibration and audio do not.

The demo phone is a sensor bridge. It does not measure the slab. The production node does.

## Pipeline

Seven stages, in order.

### 1. High-pass **2 Hz** (runbook heuristic)

Removes DC and gravity only. It is not a noise canceller. Walking, TV vibration, and impacts still pass.

### 2. Trigger

Short-term energy \(R\) is compared to a quiet-floor baseline:

\[
\theta = \mathrm{median}(R) + k \cdot \mathrm{MAD}(R), \quad k = 6
\]

**k = 6** is a runbook heuristic. Scaled MAD is approximately \(\sigma\) if the quiet floor is Gaussian (physics). Fire only if the peak also exceeds **min_peak_mag = 3 m/s²** (runbook heuristic). The baseline updates on quiet samples only. After a fire, refractory **2 s** (runbook heuristic). No threshold until **5** quiet samples have been seen (runbook heuristic); until then the trigger is in warmup.

This stage answers one question: did something hit the floor? It does not classify the hit.

### 3. Window

Wait **post_s = 1.5 s** (runbook heuristic), then clip **pre_s = 0.5 s** plus that **1.5 s**. The **2 s** length is an aperture so the decay is inside the clip. It is not a claim that falls last 2 s.

Our fig. 6 caption draws a body decaying over about **1 s** and a rigid object stopping within about **150 ms**. Those times are **unmeasured** here and are **not cited**. Do not attribute them to a paper.

Alwan et al. (2006) is cited only for useful range of about **15–20 ft** on **concrete** (**Alwan range**). That number does not transfer to tile, vinyl, laminate, or carpet, and it is not a range this build has measured.

### 4. Seven features on \(|w|\)

\(w\) is the high-passed acceleration magnitude in the clip. Features exist to describe **shape**, not only amplitude. Peak amplitude of a body and a dropped object can be similar, so amplitude alone does not separate them.

| Feature | What it measures | Why it exists |
|---|---|---|
| `peak` | max \(\lvert w \rvert\) | Amplitude of the hit |
| `rms` | energy of the clip | Scale for crest; a loud short click and a quieter long thud can share a peak |
| `crest` | peak / rms | Shape: impulsive (high) vs energy spread in time (lower) |
| `rise_ms` | time from 10% of peak up to the peak (runbook heuristic) | Onset; a slow sit-down is not a strike |
| `decay_ms` | time from the peak down to 10% of peak (runbook heuristic) | Duration of the tail; the main shape cue in this build |
| `low_ratio` | fraction of spectral energy at low frequency | Body-like energy sits lower than a rigid click, **when** the bandwidth exists |
| `centroid_hz` | spectral centre of mass | Same job as `low_ratio`, as a single frequency |

Sampling is **50 Hz** (runbook heuristic). Nyquist is **25 Hz** (physics). Spectral features (`low_ratio`, `centroid_hz`) are weak on the phone. Envelope features (`peak`, `rms`, `crest`, `rise_ms`, `decay_ms`) may still separate a bag from a book on hard tile. That separation is **empirical** and currently **unmeasured**. If histograms on this floor overlap, the features do not work here. Say so.

### 5. Score \(p \in [0,1]\)

`RuleClassifier` is the default. `ModelClassifier` is used if `models/clf.joblib` exists. **\(p^* = 0.60\)** is a **placeholder** until a precision–recall sweep on labelled CSVs. If \(p < p^*\), the event stays a candidate and the confirm window does not start. If \(p \ge p^*\), the audio check may still suppress it. Telegram is sent only after confirm times out to **alert**.

### 6. Audio check

A second classifier may suppress the alarm. It is fail-open: if it is off, broken, or has no usable audio, the kinematic score stands. YAMNet is optional and expects **16 kHz** PCM (YAMNet required input; **unmeasured** on this phone path). The phone does not send 16 kHz PCM; it sends a scalar dB. This stage does **not** require speech after a fall. A person who cannot call is still a candidate for alert.

### 7. Confirm, then escalate

After a candidate that passes the score and is not suppressed by audio:

- Demo window **10 s** (runbook heuristic). Product window **60 s** (runbook heuristic).
- Motion during the window → **recovered**.
- Cancel on the dashboard → **cancelled**.
- Timeout with no recovery and no cancel → **alert** once.

Only **alert** sends Telegram (room, time, score). Rung 2 at **60 s** (runbook heuristic) unless acknowledged. `dry_run` defaults **true** (runbook heuristic); nothing leaves the laptop until that flag is flipped.

## Learning

The feature extractor stays frozen. Fit only the last layer on this floor, from about **10** heel-drops and about **10** object drops (runbook heuristic; see `docs/data-collection.md`). Robustness is empirical: it holds if the histograms on this floor separate, and not otherwise.

Report precision, recall, F1, and a confusion matrix. One event per file. Do not report accuracy. Falls are rare; a constant “no fall” predictor scores high and is useless.

Errors always exist. The product question is rates: misses versus false alarms per month. Those rates are **unmeasured**.

## Distance and coverage

One sensor cannot range. Amplitude decays with distance \(r\) (physics). A large peak nearby and a smaller peak far away are not separable from amplitude. Coverage is measured at install, room by room, not assumed from a datasheet. Alwan’s **15–20 ft** on concrete (**Alwan range**) is a cited bound on one construction, not a specification for this unit.

## Hyperparameters

Set these from labelled CSVs after you have histograms. Do not tune them live on the demo until that is done.

| Name | Current value | Label | How to set from labelled CSVs |
|---|---|---|---|
| `k` | 6 | runbook heuristic | Histogram quiet-floor RMS. Raise if walk/quiet still trigger. Lower if heel-drop never crosses \(\theta\). If you need \(k < 3\) (runbook heuristic) to catch positives, the recordings are too weak; move the phone closer and re-record. |
| `min_peak_mag` | 3 m/s² | runbook heuristic | Absolute floor. Set above walk peaks and below heel-drop / bag peaks on this floor. |
| `pre_s` | 0.5 s | runbook heuristic | Enough quiet lead-in that rise_ms is not clipped. |
| `post_s` | 1.5 s | runbook heuristic | Enough tail that decay_ms is not truncated. If decay histograms pile up at the window length, lengthen `post_s` and re-extract. |
| \(p^*\) | 0.60 | placeholder | Sweep on held-out labelled files. Plot precision–recall. Pick a threshold; write it into `config.yaml`. Until that sweep exists, 0.60 is a placeholder, not a result. |

## Work sequence

1. **Build** the pipeline (this repo).
2. **Collect** labelled CSVs (`docs/data-collection.md`). One event per file.
3. **Analyze** feature histograms on this floor, especially `decay_ms`, `crest`, `low_ratio`. If classes overlap, stop claiming they separate.
4. **Train** the last layer (or fit rule references) on those files. Write `models/clf.joblib`.
5. **Test** on unseen takes. Heel-drop / bag → alert or recovered. Book / walk / quiet → no Telegram.

Direction of impact is not required for a binary fall / not-fall proof of concept.
