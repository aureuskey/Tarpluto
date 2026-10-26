# Tarpluto - sensor sources for the Neural Patch (MyoWare 2.0 envelope, 0-1023)
import collections
import threading
import time

import numpy as np

WINDOW_S = 1.0  # signal = peak in the last second before Enter


class Stream:
    """Background reader: keeps (time, raw) samples; calibrates rest -> max flex."""

    def __init__(self):
        self.buf = collections.deque(maxlen=2000)
        self.lo, self.hi = 0.0, 1023.0
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            v = self.sample()
            if v is not None:
                self.buf.append((time.time(), v))

    def _level(self, seconds):
        t0 = time.time()
        time.sleep(seconds)
        vals = [v for t, v in list(self.buf) if t >= t0]
        return float(np.median(vals)) if vals else 0.0

    def calibrate(self):
        input("[Calibrate] Relax your muscle, then press Enter (2 s)...")
        self.calibrating = True
        self.lo = self._level(2.0)
        input("[Calibrate] Flex as hard as you can, then press Enter (2 s)...")
        self.flexing = True
        self.hi = max(self._level(2.0), self.lo + 1.0)
        self.flexing = self.calibrating = False
        self.buf.clear()  # don't let the calibration flex leak into the first thought
        print(f"[Calibrate] rest {self.lo:.0f} | max {self.hi:.0f}\n")

    def peak(self):
        now = time.time()
        vals = [v for t, v in list(self.buf) if t >= now - WINDOW_S]
        raw = max(vals) if vals else self.lo
        return float(np.clip((raw - self.lo) / (self.hi - self.lo), 0.0, 1.0))


class SerialStream(Stream):
    def __init__(self, port, baud):
        try:
            import serial  # optional: pip install pyserial
        except ImportError:
            raise SystemExit("--source serial needs pyserial: pip install pyserial")
        self.ser = serial.Serial(port, baud, timeout=1)
        super().__init__()

    def sample(self):
        try:
            return float(self.ser.readline().strip())
        except ValueError:
            return None


class FakeStream(Stream):
    """Simulated MyoWare: ~100 Hz, noise at rest, random flex bursts."""

    def __init__(self):
        self.flexing = self.calibrating = False
        self.burst_until, self.rng = 0.0, np.random.default_rng()
        super().__init__()

    def sample(self):
        time.sleep(0.01)
        now = time.time()
        if not self.calibrating and now > self.burst_until and self.rng.random() < 0.01:
            self.burst_until, self.level = now + self.rng.uniform(0.3, 0.8), self.rng.uniform(150, 650)
        v = 80 + self.rng.normal() * 10
        if self.flexing:
            v += 650
        elif now < self.burst_until:
            v += self.level
        return float(np.clip(v, 0, 1023))


class SensorPatch:
    """Neural Patch backed by a real/fake sensor stream instead of think-time."""

    def __init__(self, stream):
        self.stream = stream
        stream.calibrate()

    def read_thought(self, thought, think_time):
        signal = self.stream.peak()
        print(f"[Neural Patch] '{thought}' | sensor peak (last {WINDOW_S:.0f}s) -> signal {signal:.2f}")
        return signal
