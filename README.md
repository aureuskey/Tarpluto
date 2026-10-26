# Tarpluto

A software simulation of a non-invasive neural interface loop:
**Neural Patch** (intent signal) → **Living Node** (1,000 simulated spiking neurons).

Today the Patch is a proxy: the time you take to think before pressing Enter becomes a signal from 0 to 1. The Living Node turns that signal into spikes, with leak, threshold, refractory period and short-term plasticity (facilitation/depression). No hardware yet.

## Run

```bash
git clone https://github.com/aureuskey/Tarpluto
cd Tarpluto
pip install -r requirements.txt && python sim/core_loop.py --demo
```

Type a few thoughts, some fast and some slow, then `quit`. Commands: `status`, `quit`.
On quit, one figure is saved to `demos/tarpluto_loop.png`: Patch signal per thought (top) and the spike raster (bottom).

Flags: `--demo` (fixed seed, fresh start, one AI line per thought), `--ai`, `--memory` (load/save state in `memory.json`), `--figure PATH`.

## Hardware (optional)

Swap think-time for a real signal with a [MyoWare 2.0](https://www.sparkfun.com/myoware-2-muscle-sensor.html) muscle sensor (muscle intent, not brain signals).

1. Wire MyoWare ENV → Arduino A0, flash `hardware/myoware_serial.ino`.
2. `pip install pyserial && python sim/core_loop.py --source serial --port /dev/ttyACM0` (Windows: `COM3`).
3. Calibrate (relax 2 s, then max flex 2 s). Flex while you think; the peak in the last second before Enter is the signal.

No hardware? `python sim/core_loop.py --source fake` runs the same loop on a simulated MyoWare stream.
