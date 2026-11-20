# Changelog

## v0.11

- Optional session-level perturbations on the synthetic EMG generator: electrode shift (channel remix and partial mixing), per-channel amplitude drift between sessions, and muscle fatigue (gradual amplitude drop and spectral / zero-crossing change across trials). With the perturbations left off, the generator matches the previous unperturbed trials.
- Cross-session eval: train on session 1, then score later simulated sessions with no recalibration and after a centroid update from 3 labelled trials per command. `python3 sim/decode_eval.py --sessions N` prints both accuracies and writes `demos/session_accuracy.png`. Synthetic data, simulated sessions. Synthetic-data numbers only. No sensor was read.
- Simulation header, banner and plot title are v0.11.

## v0.10

- Synthetic 4-channel jaw/face EMG (fixed seed, sensor noise, 50 Hz hum, per-trial gain, time jitter, crosstalk, baseline drift) and a nearest-centroid decoder for yes, no, help, water, stop.
- Windowed per-channel features: RMS, mean absolute value, zero crossings, waveform length.
- `python3 sim/decode_eval.py` prints held-out accuracy and a confusion matrix and writes `demos/confusion_matrix.png`. Synthetic-data numbers only. No sensor was read.
- `--decode` and `--source fake4ch` on `sim/core_loop.py`. A decoded command acts only after you type `y`.
- Simulation header, banner and plot title are v0.10.

## v0.9

- Annotated tag `v0.9` on `dd1d152` (13 Jul 2026 README status).
