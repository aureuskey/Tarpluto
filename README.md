# Tarpluto

A software simulation of a non-invasive neural interface loop:
**Neural Patch** (intent signal) → **Living Node** (1,000 simulated spiking neurons).

Today the Patch is a proxy: the time you take to think before pressing Enter becomes a signal from 0 to 1. The Living Node turns that signal into spikes, with leak, threshold, refractory period and short-term plasticity (facilitation/depression). No hardware yet.

## Status (v0.10)

Software simulation. `python3 sim/decode_eval.py` prints synthetic-data numbers only. No sensor was read. This repo does not report hardware or real-world accuracy.

## Run

```bash
git clone https://github.com/aureuskey/Tarpluto
cd Tarpluto
pip install -r requirements.txt && python3 sim/core_loop.py --demo
```

Type a few thoughts, some fast and some slow, then `quit`. Commands: `status`, `quit`.
On quit, one figure is saved to `demos/tarpluto_loop.png`: Patch signal per thought (top) and the spike raster (bottom).

Flags: `--demo` (fixed seed, fresh start, one AI line per thought), `--ai`, `--memory` (load/save state in `memory.json`), `--figure PATH`. `--demo` alone does not decode.

## Silent-command decoder (synthetic)

Five labels: `yes`, `no`, `help`, `water`, `stop`. `SyntheticJawEMG` in `sim/sensor.py` builds a 4-channel jaw/face EMG trial for each label (fixed seed, sensor noise, 50 Hz hum, per-trial gain, time jitter, electrode crosstalk, baseline drift). `sim/decode.py` cuts the trial into windows and, per channel, computes RMS, mean absolute value, zero crossings and waveform length. A nearest-centroid classifier maps that vector to a label.

Held-out eval, from the repo root:

```bash
python3 sim/decode_eval.py
```

Prints accuracy and a confusion matrix, and writes `demos/confusion_matrix.png`. The checked-in PNG is the output of that command. Synthetic-data numbers only. No sensor was read. A diagonal matrix means these invented patterns were separable at the current noise settings. It is not a device result.

The decoder stays off unless you ask for it. A predicted label does nothing until you type `y`:

```bash
python3 sim/core_loop.py --demo --decode
python3 sim/core_loop.py --source fake4ch
```

Type one of the five labels to synthesize that pattern, read the prediction, then `y` to let the predicted command drive the Living Node. Any other reply drops it. Training uses seed 7 (40 trials per label). Each demo session draws further synthetic trials from seed 8. `python3 sim/core_loop.py --demo` does not decode.

Tests: `python3 sim/test_decode.py`

## Hardware (optional)

Swap think-time for a real signal with a [MyoWare 2.0](https://www.sparkfun.com/myoware-2-muscle-sensor.html) muscle sensor (muscle intent, not brain signals).

1. Wire MyoWare ENV → Arduino A0, flash `hardware/myoware_serial.ino`.
2. `pip install pyserial && python sim/core_loop.py --source serial --port /dev/ttyACM0` (Windows: `COM3`).
3. Calibrate (relax 2 s, then max flex 2 s). Flex while you think; the peak in the last second before Enter is the signal.

No hardware? `python sim/core_loop.py --source fake` runs the same loop on a simulated MyoWare stream.

## Docs

- [Roadmap](docs/roadmap.md)
- [Changelog](CHANGELOG.md)
