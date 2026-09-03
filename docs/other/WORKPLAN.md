# OPOYO PoC work sequence

1. **Build the pipeline** (code). Trigger → 2 s window → 7 features + rule/CNN score → audio check → 10 s confirm → Telegram. Recorder writes labelled CSVs from the phone stream.
2. **Collect labelled clips** (you / friend). Procedure in `docs/data-collection.md`. Output: `data/<label>_<nn>.csv` plus one `metadata.csv`.
3. **Analyze and train** (model side). Plot feature histograms (especially `decay_ms`, `crest`, `low_ratio`) on this floor. If classes overlap, say so and stop pretending the features work. If they separate: fit rule refs and/or last-layer probe; write `models/` and a confusion matrix.
4. **Test** (you). Unseen drops, same phone setup. Bag/heel-drop → `alert` (or `recovered` if you stand up). Book/walk/quiet → no Telegram.

Do not tune thresholds by poking the live demo until step 3 has numbers.

**What we are claiming, and what we are not.** Direction of impact is not required for a binary fall/not-fall PoC. High-frequency spectrum is the paper’s story; the phone at 50 Hz mostly sees envelope (peak, RMS, crest, rise, decay). That can still separate a heavy damped hit from a click **on hard tile**, if the histograms say so. That uniqueness is not guaranteed. Step 3 is the test of the theory, not a formality.
