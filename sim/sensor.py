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
    """Simulated MyoWare: ~100 Hz, noise at rest, random flex bursts.

    One channel. SyntheticJawEMG below is the 4-channel jaw/face stand-in.
    """

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


# --- Synthetic 4-channel jaw/face EMG ---------------------------------------
# Invented patterns for a software decoder. Not a recording. Every knob below
# is a fixed parameter; EMG_SEED makes a run repeatable.

FS = 200
TRIAL_S = 0.80
N_SAMPLES = int(round(TRIAL_S * FS))
N_CHANNELS = 4
CHANNEL_NAMES = ('left_masseter', 'right_masseter', 'submental', 'jaw_strip')
SILENT_COMMANDS = ('yes', 'no', 'help', 'water', 'stop')
EMG_SEED = 7
ONLINE_SEED = EMG_SEED + 1

NOISE_STD = 4.0          # additive sensor noise, synthetic ADC counts
HUM_HZ = 50.0
HUM_STD = 1.5            # per-channel amplitude of a 50 Hz hum
DRIFT_STD = 5.0          # end-of-trial baseline slope, per channel
GAIN_STD = 0.12          # per-channel multiplicative jitter
TRIAL_SCALE_STD = 0.05   # one gain shared by all channels in a trial
CROSSTALK = 0.10         # fraction of each channel mixed in from the others
MAX_SHIFT = int(round(0.03 * FS))  # ±30 ms envelope shift

# Bumps: (channel, center_s, sigma_s, peak amplitude). Peak amplitude is the
# standard deviation of the noise carrier at the center of the bump.
TEMPLATES = {
    'yes': (
        (0, 0.30, 0.06, 75.0),
        (1, 0.30, 0.06, 75.0),
        (2, 0.32, 0.07, 25.0),
    ),
    'no': (
        (2, 0.22, 0.05, 90.0),
        (3, 0.30, 0.06, 30.0),
        (2, 0.52, 0.05, 80.0),
    ),
    'help': (
        (3, 0.16, 0.05, 95.0),
        (0, 0.48, 0.07, 60.0),
        (2, 0.58, 0.06, 30.0),
    ),
    'water': (
        (2, 0.16, 0.05, 50.0),
        (3, 0.34, 0.06, 85.0),
        (1, 0.40, 0.08, 25.0),
        (2, 0.56, 0.05, 70.0),
    ),
    'stop': (
        (0, 0.50, 0.035, 130.0),
        (1, 0.50, 0.035, 130.0),
        (3, 0.50, 0.04, 40.0),
    ),
}


def _mix_matrix():
    """Electrode mix: each channel keeps (1 - CROSSTALK) and shares the rest."""
    off = CROSSTALK / (N_CHANNELS - 1)
    mix = np.full((N_CHANNELS, N_CHANNELS), off)
    np.fill_diagonal(mix, 1.0 - CROSSTALK)
    return mix


MIX = _mix_matrix()


def _check_templates():
    if tuple(TEMPLATES) != SILENT_COMMANDS:
        raise RuntimeError('TEMPLATES keys must match SILENT_COMMANDS, in order')
    for name, bumps in TEMPLATES.items():
        if not bumps:
            raise RuntimeError(f'{name} has no bumps')
        for ch, center, sigma, amp in bumps:
            if not 0 <= ch < N_CHANNELS:
                raise RuntimeError(f'{name} channel {ch}')
            if not 0.0 < center < TRIAL_S:
                raise RuntimeError(f'{name} center {center}')
            if sigma <= 0 or amp <= 0:
                raise RuntimeError(f'{name} bump is not positive')


_check_templates()


class SyntheticJawEMG:
    """4-channel synthetic jaw/face EMG trials, one mouthed command each.

    The waveform is amplitude-modulated noise plus sensor noise, hum, crosstalk
    and a linear baseline drift. Labels select TEMPLATES. Nothing here was
    measured from a sensor.
    """

    def __init__(self, seed=EMG_SEED):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def trial(self, command):
        if command not in TEMPLATES:
            names = ', '.join(SILENT_COMMANDS)
            raise ValueError(f'unknown command {command!r}; synthetic commands are {names}')
        t = np.arange(N_SAMPLES) / FS
        env = np.zeros((N_SAMPLES, N_CHANNELS))
        for ch, center, sigma, amp in TEMPLATES[command]:
            env[:, ch] += amp * np.exp(-0.5 * ((t - center) / sigma) ** 2)

        shift = int(self.rng.integers(-MAX_SHIFT, MAX_SHIFT + 1))
        if shift:
            env = np.roll(env, shift, axis=0)
            if shift > 0:
                env[:shift] = 0.0
            else:
                env[shift:] = 0.0

        trial_gain = float(np.clip(self.rng.normal(1.0, TRIAL_SCALE_STD), 0.7, 1.3))
        ch_gain = np.clip(self.rng.normal(1.0, GAIN_STD, size=N_CHANNELS), 0.45, 1.7)
        env = env * (trial_gain * ch_gain)

        emg = env * self.rng.normal(0.0, 1.0, size=env.shape)
        emg = emg + self.rng.normal(0.0, NOISE_STD, size=emg.shape)
        emg = emg @ MIX
        slope = self.rng.normal(0.0, DRIFT_STD, size=N_CHANNELS)
        emg = emg + np.linspace(0.0, 1.0, N_SAMPLES)[:, None] * slope
        hum = self.rng.normal(0.0, HUM_STD, size=N_CHANNELS)
        emg = emg + np.sin(2.0 * np.pi * HUM_HZ * t)[:, None] * hum
        return emg

    def collect(self, n_per_class, commands=SILENT_COMMANDS):
        """Trials grouped by command, in `commands` order. One RNG stream."""
        if n_per_class < 1:
            raise ValueError('n_per_class must be >= 1')
        trials, labels = [], []
        for command in commands:
            for _ in range(n_per_class):
                trials.append(self.trial(command))
                labels.append(command)
        return trials, labels
