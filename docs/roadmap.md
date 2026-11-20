# Roadmap

## Status (v0.11)

Tarpluto is a laptop simulation of a non-invasive neural-interface loop.

- Neural Patch to Living Node: `python3 sim/core_loop.py --demo`. The Patch signal is think-time, or an optional one-channel MyoWare stand-in (`--source fake` or `--source serial`).
- Synthetic 4-channel jaw/face EMG and a nearest-centroid decoder for five command labels: `python3 sim/decode_eval.py`, or `--decode` / `--source fake4ch` in the loop. Synthetic-data numbers only. No sensor was read.
- Cross-session synthetic eval: `python3 sim/decode_eval.py --sessions 4`. Later sessions add electrode shift, per-channel amplitude drift and muscle fatigue, then score accuracy with no recalibration and after a short centroid update. Synthetic data, simulated sessions.

This repo does not report hardware or real-world accuracy.

## Next

Record 2–4 channel surface EMG of 5–10 silently mouthed commands and run the same features and a held-out matrix on that recording. Keep the confirm step (type `y`) in front of any command that acts. That recording is not in this repo. Until it is, the numbers to quote are the output of `sim/decode_eval.py` and `sim/decode_eval.py --sessions`.

## Later

- Keep the software path small: windowed features, a centroid or linear classifier, a held-out matrix, confirm-before-act.
- Leave hardware out of the accuracy table until a recording is what the eval script reads.
