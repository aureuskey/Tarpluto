# Tarpluto - core simulation loop
# Neural Patch (think-time or MyoWare sensor -> signal) -> Living Node (1000 spiking neurons) -> output
import argparse
import json
import os
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

N_NEURONS = 1000
MEMORY_FILE = 'memory.json'
FIGURE_FILE = os.path.join('demos', 'tarpluto_loop.png')


class NeuralPatch:
    """Proxy for a neural sensor: longer think-time = stronger intent signal (0-1)."""

    def read_thought(self, thought: str, think_time: float) -> float:
        if think_time < 0.2:
            signal = think_time * 2.5                    # 0-0.2 s -> 0-0.5
        else:
            signal = 0.5 + (think_time - 0.2) * 0.25     # 0.2 s+ -> 0.5-1.0
        signal = float(np.clip(signal + np.random.uniform(-0.02, 0.02), 0.0, 1.0))
        print(f"[Neural Patch] '{thought}' | think-time {think_time:.2f}s -> signal {signal:.2f}")
        return signal


class LivingNode:
    """1000-neuron model with leak, spike threshold, refractory period and
    short-term plasticity (facilitation/depression)."""

    def __init__(self, neural_state=None, adaptation=0.0):
        self.neural_state = np.zeros(N_NEURONS) if neural_state is None else np.array(neural_state, float)
        self.adaptation = adaptation
        self.refractory = np.zeros(N_NEURONS)
        self.facilitation = 0.0
        self.depression = 0.0
        self.signals = []
        self.spike_events = []

    def process(self, signal: float) -> float:
        plasticity = float(np.clip(1.0 + self.facilitation - self.depression, 0.5, 2.0))
        drive = signal * np.abs(np.random.randn(N_NEURONS)) * 0.35 * plasticity
        self.neural_state = np.clip(self.neural_state * 0.9 + drive, 0.0, 1.0)
        self.refractory = np.maximum(self.refractory - 1, 0)

        threshold = 0.55 + 0.1 * self.adaptation
        spikes = np.where((self.neural_state > threshold) & (self.refractory == 0))[0]
        step = len(self.signals)
        self.spike_events.extend((step, int(i)) for i in spikes)
        if len(spikes):
            self.neural_state[spikes] = 0.0
            self.refractory[spikes] = 2
            self.facilitation += 0.04
            self.depression += 0.025
        self.facilitation *= 0.95
        self.depression *= 0.90
        self.adaptation = float(np.clip(self.adaptation + 0.004 * signal, 0.0, 2.0))
        self.signals.append(signal)

        output = float(np.mean(self.neural_state) * (1.0 + self.adaptation) + 0.015 * len(spikes) / 10)
        print(f"[Living Node] spikes {len(spikes)} | facilitation {self.facilitation:.2f} | "
              f"depression {self.depression:.2f} -> output {output:.2f}")
        return output, len(spikes)


def ai_line(signal: float, spikes: int) -> str:
    """One plain sentence describing what the Node did with the Patch signal."""
    level = 'strong' if signal >= 0.7 else 'moderate' if signal >= 0.3 else 'weak'
    return f"{level.capitalize()} signal ({signal:.2f}) fired {spikes} of {N_NEURONS} neurons."


def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return None, 0.0
    try:
        with open(MEMORY_FILE) as f:
            data = json.load(f)
        state = data.get('neural_state')
        if not state or len(state) != N_NEURONS:
            print(f"[Memory] Ignored {MEMORY_FILE}: shape does not match {N_NEURONS} neurons.")
            return None, 0.0
        print("[Memory] Loaded previous state.")
        return state, float(data.get('adaptation', 0.0))
    except Exception as e:
        print(f"[Memory] Failed to load: {e}")
        return None, 0.0


def save_memory(node):
    with open(MEMORY_FILE, 'w') as f:
        json.dump({'neural_state': node.neural_state.tolist(), 'adaptation': node.adaptation}, f)
    print(f"[Memory] Saved state to {MEMORY_FILE}.")


def plot_loop(node, path=FIGURE_FILE, show=True):
    """Top: Patch signal per thought. Bottom: Node spike raster."""
    if not node.signals:
        print("[Plot] Nothing to plot.")
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    steps = range(len(node.signals))
    ax1.plot(steps, node.signals, 'o-')
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel('Patch signal')
    ax1.set_title('Tarpluto: Neural Patch signal -> Living Node spikes')
    if node.spike_events:
        s, n = zip(*node.spike_events)
        ax2.scatter(s, n, s=2, c='black', marker='.')
    ax2.set_ylim(-0.5, N_NEURONS - 0.5)
    ax2.set_ylabel('Neuron index')
    ax2.set_xlabel('Thought #')
    ax2.set_xticks(list(steps))
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"[Plot] Saved figure to {path}")
    if show and matplotlib.get_backend().lower() != 'agg':
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Tarpluto Patch -> Node simulation')
    ap.add_argument('--demo', action='store_true', help='fixed seed, fresh 1000-neuron start')
    ap.add_argument('--memory', action='store_true', help=f'load/save state in {MEMORY_FILE}')
    ap.add_argument('--ai', action='store_true', help='print one AI line per thought (on in --demo)')
    ap.add_argument('--figure', default=FIGURE_FILE, help='where to save the figure')
    ap.add_argument('--source', choices=['thinktime', 'serial', 'fake'], default='thinktime',
                    help='Patch signal source (serial = MyoWare via Arduino)')
    ap.add_argument('--port', help='serial port, e.g. /dev/ttyACM0 or COM3')
    ap.add_argument('--baud', type=int, default=115200)
    args = ap.parse_args()
    use_memory = args.memory and not args.demo
    if args.demo:
        np.random.seed(42)

    state, adaptation = load_memory() if use_memory else (None, 0.0)
    node = LivingNode(state, adaptation)
    if args.source == 'thinktime':
        patch = NeuralPatch()
    else:
        from sensor import SensorPatch, SerialStream, FakeStream
        if args.source == 'serial' and not args.port:
            ap.error('--source serial needs --port')
        stream = SerialStream(args.port, args.baud) if args.source == 'serial' else FakeStream()
        patch = SensorPatch(stream)

    print("\n=== Tarpluto ===")
    if args.source == 'thinktime':
        print("Type a thought. Take longer to think -> stronger signal -> more spikes.")
    else:
        print("Type a thought, flex while you think, press Enter. Peak flex in the last second = signal.")
    print("Commands: status, quit\n")
    try:
        while True:
            start = time.time()
            try:
                text = input("> ").strip()
            except EOFError:
                break
            think_time = time.time() - start
            if text.lower() == 'quit':
                break
            if text.lower() == 'status':
                print(f"[Status] thoughts {len(node.signals)} | total spikes {len(node.spike_events)} | "
                      f"adaptation {node.adaptation:.3f}")
                continue
            if not text:
                continue
            signal = patch.read_thought(text, think_time)
            _, spikes = node.process(signal)
            if args.ai or args.demo:
                print(f"[AI] {ai_line(signal, spikes)}")
            print()
    except KeyboardInterrupt:
        print()
    plot_loop(node, args.figure)
    if use_memory:
        save_memory(node)


if __name__ == '__main__':
    main()
