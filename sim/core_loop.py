# Tarpluto - Core Simulation Loop v0.4
# Neural Patch (input) -> Living Node (wetware sim) -> Output
# Features:
#   1. NeuralPatch measures typing/think time as proxy for thought intensity
#   2. Symbiotic AI gives creative suggestion + practical next step
#   3. Simple command system: status, save, fatigue, help, quit
#   4. Auto‑save memory (JSON) and plot (PNG) on exit or on "save"

import numpy as np
import time
import matplotlib.pyplot as plt
import json
import os
import random
import sys
import threading
from datetime import datetime

MEMORY_FILE = 'memory.json'
PLOT_FILE   = os.path.join('demos', 'simulation_plot.png')

# ----------------------------------------------------------------------
# Neural Patch – derives signal strength from typing/think duration
# think time and optional Enter‑hold
# ----------------------------------------------------------------------
class NeuralPatch:
    def __init__(self):
        self.signal_strength = 0.0
        self._last_key_time = time.time()
        self._hold_start = None   # timestamp when Enter was first held down
        self._hold_active = False

    def _update_timing(self):
        """Update internal timers based on key press/release events via a simple hook."""
        # We'll approximate by measuring time between successive input() calls.
        now = time.time()
        delta = now - self._last_key_time
        self._last_key_time = now
        return delta

    def read_thought(self, intent: str) -> float:
        """
        Measure the time the user spent before submitting the input.
        If Enter is held down (detected by an empty line followed quickly by another),
        we increase the signal proportionally to the hold duration.
        Returns a signal strength in [0,1].
        """
        think_time = self._update_timing()

        # Detect a potential Enter‑hold: if the user just pressed Enter on an empty line
        # and the previous input was also empty (or we are already holding).
        if intent == "" and not self._hold_active:
            # start hold timer
            self._hold_start = time.time()
            self._hold_active = True
            # treat as a short think time for now; will be extended on release
            think_time = 0.0
        elif intent == "" and self._hold_active:
            # still holding; update think_time based on how long Enter has been held
            think_time = time.time() - self._hold_start
        else:
            # any non‑empty input releases the hold
            if self._hold_active:
                hold_duration = time.time() - self._hold_start
                think_time += hold_duration   # add the hold time to the think time
                self._hold_active = False
                self._hold_start = None

        # Map think_time to signal: 0‑2 s maps linearly to 0‑1, with jitter
        base = min(1.0, max(0.0, think_time / 2.0))
        jitter = random.uniform(-0.03, 0.03)
        self.signal_strength = np.clip(base + jitter, 0.0, 1.0)

        status = "HOLD" if self._hold_active else ("EMPTY" if intent == "" else "NORMAL")
        print(f"[Neural Patch] Thought: '{intent if intent else '(hold Enter)'}' | "
              f"Think‑time: {think_time:.2f}s ({status}) → Signal: {self.signal_strength:.2f}")
        return self.signal_strength

# ----------------------------------------------------------------------
# Living Node – wet‑ware style processing with memory & history
# ----------------------------------------------------------------------
class LivingNode:
    def __init__(self, neural_state=None, adaptation=0.0):
        if neural_state is None:
            self.neural_state = np.zeros(400)        # larger neural pool for richer dynamics
        else:
            self.neural_state = np.array(neural_state, dtype=float)
        self.adaptation = adaptation                 # plasticity factor
        self.history = []                            # for visualization (store mean activity)
        self.refractory = np.zeros_like(self.neural_state)  # simple refractory counter

    def process(self, input_signal: float) -> float:
        """Biological integration, plasticity, refractory, and spike‑like output."""
        # 1. Add noisy drive
        drive = input_signal * np.random.randn(*self.neural_state.shape) * 0.25

        # 2. Update state with leaky integration
        self.neural_state = np.clip(self.neural_state + drive - 0.01 * self.neural_state, 0.0, 1.0)

        # 3. Refractory: after a spike, suppress, a neuron cannot fire for a few steps
        self.refractory = np.maximum(self.refractory - 1, 0)

        # 4. Detect spikes where activity crosses a dynamic threshold
        spike_threshold = 0.6 + 0.1 * self.adaptation  # threshold rises with learning
        spikes = (self.neural_state > spike_threshold) & (self.refractory == 0)
        spike_count = np.sum(spikes)

        # 5. Refractory reset for spiking neurons
        self.neural_state[spikes] = 0.0
        self.refractory[spikes] = 5   # refractory period in steps

        # 6. Plasticity: strengthen active synapses proportional to input
        self.adaptation += 0.005 * input_signal  # slower learning
        self.adaptation = np.clip(self.adaptation, 0.0, 2.0)

        # 7. Record mean activity for plotting
        self.history.append(np.mean(self.neural_state))

        # 8. Output: combine mean activity with adaptation and spike count
        output = np.mean(self.neural_state) * (1.0 + self.adaptation) + 0.01 * spike_count
        print(f"[Living Node] Adaptation: {self.adaptation:.2f} | "
              f"Mean activity: {np.mean(self.neural_state):.3f} | "
              f"Spikes: {spike_count} → Output: {output:.2f}")
        return output

# ----------------------------------------------------------------------
# Symbiotic AI – creative suggestion + practical next step
# ----------------------------------------------------------------------
def symbiotic_ai(user_input: str, node_output: float) -> str:
    """
    Return a two‑part response:
    1. Imaginative spark (metaphor, insight)
    2. Concrete next step (actionable suggestion)
    Uses simple heuristics; can be swapped for a local LLM later.
    """
    # Seed randomness with node output for reproducibility within a session
    seed_val = int((node_output * 1e6) % 2**32)
    local_random = random.Random(seed_val)

    # ---- Creative spark ----
    sparks = [
        "A quiet spark flickers in the substrate, hinting at a hidden pattern.",
        "Your thought ripples through the wet‑ware like a stone in a still pond.",
        "The neural lattice hums, resonating with a frequency just beyond awareness.",
        "A faint glow appears at the edge of the cortical map—something new is forming.",
        "Echoes of your intent bounce between dendrites, seeking a path forward.",
        "The bio‑digital interface tingles, as if a synapse just fired for the first time.",
        "Patterns emerge from the noise, like constellations appearing in a cloudy sky.",
        "A whisper of understanding passes through the glial network.",
        "The membrane potential shifts, preparing for an action potential of insight.",
        "Silence breaks, and a micro‑burst of activity cascades across the network."
    ]
    spark = local_random.choice(sparks)

    # ---- Practical next step ----
    # Base suggestions on length of input and output magnitude
    if len(user_input.strip()) == 0:
        action = "Try holding Enter for a few seconds to build a sustained signal."
    elif node_output < 0.3:
        action = "Increase focus: speak your intention aloud before typing."
    elif node_output < 0.6:
        action = "Break the idea into smaller steps and tackle the first one now."
    else:
        action = "You're in a high‑flow state—capture this insight in a quick note or sketch."

    return f"{spark}\n💡 Action: {action}"

# ----------------------------------------------------------------------
# Persistence helpers
# ----------------------------------------------------------------------
def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r') as f:
                data = json.load(f)
            neural_state = data.get('neural_state')
            adaptation = float(data.get('adaptation', 0.0))
            print(f"[Memory] Loaded previous state. Adaptation: {adaptation:.3f}")
            return neural_state, adaptation
        except Exception as e:
            print(f"[Memory] Failed to load: {e}")
    return None, 0.0

def save_memory(neural_state, adaptation):
    try:
        data = {
            'neural_state': [float(x) for x in neural_state],
            'adaptation': float(adaptation)
        }
        with open(MEMORY_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[Memory] Saved state. Adaptation: {adaptation:.3f}")
    except Exception as e:
        print(f"[Memory] Failed to save: {e}")

# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_neural_states(history, save_path=PLOT_FILE):
    if not history:
        print("[Plot] No history to visualize.")
        return

    # Convert history (list of mean activity per step) to a 2D array for heatmap-ish view
    # We'll repeat the 1D signal across a few rows to get a semblance of a raster.
    data = np.array(history)
    # Create a 20‑row representation by repeating and adding slight noise for texture
    matrix = np.tile(data, (20, 1)) + np.random.normal(0, 0.01, (20, len(data)))

    plt.figure(figsize=(10, 4))
    plt.imshow(matrix.T, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Mean Neural Activity')
    plt.title('Tarpluto v0.4 – Neural Activity Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Neuron Layer (synthetic)')
    plt.tight_layout()
    os.makedirs('demos', exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved visualization to {save_path}")

# ----------------------------------------------------------------------
# Main interaction loop with command handling
# ----------------------------------------------------------------------
def main():
    print("=== Tarpluto Simulation v0.4 ===\n")
    print("Commands: status, save, fatigue, help, quit")
    print("Enter a thought/idea, or hold Enter to build sustained signal.\n")

    neural_state, adaptation = load_memory()
    patch = NeuralPatch()
    node = LivingNode(neural_state=neural_state, adaptation=adaptation)

    try:
        while True:
            # Prompt and capture raw line; we need to measure think time *before* input() returns.
            # We'll start a timer, wait for input, then compute elapsed.
            start = time.time()
            user_input = input("> ").rstrip('\n')
            think_time = time.time() - start

            # Handle special commands (case‑insensitive, stripped)
            cmd = user_input.strip().lower()
            if cmd == 'quit':
                break
            elif cmd == 'help':
                print("Available commands:")
                print("  status  – show current adaptation & recent activity")
                print("  save    – force save memory & plot now")
                print("  fatigue – simulate fatigue by temporarily reducing adaptation")
                print("  help    – show this help")
                print("  quit    – exit and save")
                print("Or type any thought/idea to process it.")
                continue
            elif cmd == 'status':
                recent = np.mean(node.history[-10:]) if len(node.history) >= 10 else (np.mean(node.history) if node.history else 0.0)
                print(f"[Status] Adaptation: {node.adaptation:.3f} | "
                      f"Recent mean activity: {recent:.3f} | "
                      f"History length: {len(node.history)} steps")
                continue
            elif cmd == 'save':
                save_memory(node.neural_state, node.adaptation)
                plot_neural_states(node.history)
                continue
            elif cmd == 'fatigue':
                fatigue_factor = 0.5
                original = node.adaptation
                node.adapter = node.adaptation * fatigue_factor  # temporary reduction
                print(f"[Fatigue] Adaptation temporarily reduced from {original:.3f} to {node.adaptation:.3f}")
                # restore after a short delay via a thread? For simplicity, just inform user it lasts until next input.
                continue

            # Normal thought processing
            # We need to inform NeuralPatch about the measured think_time.
            # Override the internal timing by injecting the measured delta.
            # We'll temporarily replace _last_key_time to reflect the measured interval.
            # Simpler: call read_thought which will compute its own delta; we already measured,
            # but we can adjust by adding the measured think_time to the internal clock.
            # To keep it simple, we will just call read_thought and ignore our measurement,
            # relying on its internal timing (which measures time between calls).
            # Since we called input() directly, the interval measured will be correct.
            signal = patch.read_thought(user_input)
            result = node.process(signal)
            advice = symbiotic_ai(user_input, result)
            print(f"[Symbiotic AI]\n{advice}\n")
            print(f"[Output] Final result: {result:.3f}\n")
            # small pause for readability
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nInterrupted!")
    finally:
        print("\nFinalizing...")
        save_memory(node.neural_state, node.adaptation)
        plot_neural_states(node.history)
        print("Remember: You are the symbiosis. Keep building.")

if __name__ == "__main__":
    main()