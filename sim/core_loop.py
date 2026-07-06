# Tarpluto - Core Simulation Loop v0.8
# Neural Patch (input) -> Living Node (wetware sim) -> Output
# Improvements:
#   1. LivingNode scaled to 1000 neurons.
#   2. Short-term synaptic plasticity (facilitation & depression).
#   3. Basic raster plot (spike timing) saved alongside heat-map.
#   4. Symbiotic AI enhanced with context-aware responses using full conversation history + node state.
#   5. Persistent memory (JSON) + auto-saved visualizations.
#   6. Live terminal status line (adaptation, burst counter, recent thoughts).
#   7. Simple learning: repeated similar thoughts boost adaptation.
#   8. Mock FinalSpark / wet-ware API wrapper.
#   9. New command "api" that sends current state to the mock API and shows realistic response.
#  10. Updated status command to show plasticity levels.
#  11. New command "brainstorm" for extended creative AI session.
#  12. Symbiotic AI now uses full conversation history and deeper node state awareness.

import numpy as np
import time
import matplotlib.pyplot as plt
import json
import os
import random
from collections import deque, defaultdict

MEMORY_FILE = 'memory.json'
HEATMAP_FILE = os.path.join('demos', 'simulation_heatmap.png')
RASTER_FILE   = os.path.join('demos', 'simulation_raster.png')

# ----------------------------------------------------------------------
# Neural Patch - derives signal strength from think-time, Enter-hold,
#                and burst detection.
# ----------------------------------------------------------------------
class NeuralPatch:
    def __init__(self):
        self.signal_strength = 0.0
        self._last_key_time = time.time()
        self._hold_start = None          # timestamp when Enter hold began
        self._hold_active = False
        self.burst_counter = 0           # decays over time, max 5
        self._recent_think_times = deque(maxlen=5)  # for burst detection

    def _update_timing(self):
        """Time elapsed since the previous input() call."""
        now = time.time()
        delta = now - self._last_key_time
        self._last_key_time = now
        return delta

    def read_thought(self, intent: str) -> float:
        """
        Convert measured think time (including possible Enter-hold)
        into a signal strength in [0,1] with burst enhancement.
        """
        think_time = self._update_timing()

        # ----- Enter-hold handling -----
        if intent == "" and not self._hold_active:
            # start of a hold
            self._hold_start = time.time()
            self._hold_active = True
            think_time = 0.0                     # will be grown on release
        elif intent == "" and self._hold_active:
            # still holding; think_time is how long Enter has been held
            think_time = time.time() - self._hold_start
        else:
            # any non-empty input releases the hold
            if self._hold_active:
                hold_duration = time.time() - self._hold_start
                think_time += hold_duration      # add the hold time
                self._hold_active = False
                self._hold_start = None

        # ----- Base signal from think time -----
        # Quick inputs (<0.2 s) get a steeper slope so they still produce a
        # noticeable signal; longer holds increase linearly but saturate at 1.
        if think_time < 0.2:
            base_signal = think_time * 2.5          # 0-0.2 s → 0-0.5
        else:
            base_signal = 0.5 + (think_time - 0.2) * 0.25  # 0.2-∞ → 0.5-1.0
        base_signal = min(1.0, max(0.0, base_signal))

        # ----- Burst detection: quick successive inputs raise burst_counter -----
        self._recent_think_times.append(think_time)
        if think_time < 0.15 and not self._hold_active:   # a quick, non-hold press
            self.burst_counter = min(self.burst_counter + 1, 5)
        else:
            self.burst_counter = max(self.burst_counter - 1, 0)  # decay

        burst_factor = 0.12 * self.burst_counter   # up to +0.60 boost
        jitter = random.uniform(-0.02, 0.02)
        signal = np.clip(base_signal + burst_factor + jitter, 0.0, 1.0)

        # ----- Store for possible external use -----
        self.signal_strength = signal

        status = ("HOLD" if self._hold_active else
                  ("BURST" if self.burst_counter > 2 else
                   ("QUICK" if think_time < 0.15 else "NORMAL")))
        print(f"[Neural Patch] Thought: '{intent if intent else '(hold Enter)'}' | "
              f"Think-time: {think_time:.2f}s ({status}) -> Signal: {signal:.2f} "
              f"(Burst: {self.burst_counter})")
        return signal

# ----------------------------------------------------------------------
# Living Node - wet-ware style processing with refractory, spike threshold,
#               and short-term synaptic plasticity (facilitation/depression).
# ----------------------------------------------------------------------
class LivingNode:
    def __init__(self, neural_state=None, adaptation=0.0):
        if neural_state is None:
            self.neural_state = np.zeros(1000)        # increased to 1000 neurons
        else:
            self.neural_state = np.array(neural_state, dtype=float)
        self.adaptation = adaptation                 # plasticity factor (0-2)
        self.history = []                            # mean activity over time (for heat-map)
        self.refractory = np.zeros_like(self.neural_state)  # refractory timer
        self._last_spike_times = deque(maxlen=10)    # simple spike history (timing)
        # Short-term plasticity variables
        self.facilitation = 0.0                      # increases with spikes, decays
        self.depression   = 0.0                      # increases with spikes, recovers
        # Spike raster storage: list of (time_step, neuron_index)
        self.spike_events = []
        # Simple learning: map thought pattern (lower-cased words) -> boost counter
        self.thought_boost = defaultdict(int)        # pattern -> count

    def process(self, input_signal: float) -> float:
        """Biological integration, plasticity, refractory, spike-like output with STP."""
        # 1. Noisy drive proportional to input, modulated by short-term plasticity
        plasticity_factor = 1.0 + self.facilitation - self.depression
        plasticity_factor = max(0.5, min(2.0, plasticity_factor))  # keep sane bounds
        drive = input_signal * np.random.randn(*self.neural_state.shape) * 0.25 * plasticity_factor

        # 2. Leaky integration (slow decay toward zero)
        self.neural_state = np.clip(
            self.neural_state + drive - 0.008 * self.neural_state,
            0.0, 1.0
        )

        # 3. Update refractory counters
        self.refractory = np.maximum(self.refractory - 1, 0)

        # 4. Dynamic spike threshold: higher adaptation → higher threshold
        spike_threshold = 0.55 + 0.1 * self.adaptation   # ≈0.55-0.75
        spikes = (self.neural_state > spike_threshold) & (self.refractory == 0)
        spike_indices = np.where(spikes)[0]
        spike_count = len(spike_indices)

        # 5. Record spikes for raster and reset spiking neurons + refractory
        if spike_count > 0:
            current_step = len(self.history)   # time step index for this iteration
            for idx in spike_indices:
                self.spike_events.append((current_step, int(idx)))
            self.neural_state[spike_indices] = 0.0
            self.refractory[spike_indices] = 6   # refractory steps
            self._last_spike_times.append(time.time())

            # Short-term plasticity updates per spike
            self.facilitation += 0.04   # facilitation increment
            self.depression   += 0.025  # depression increment

        # 6. Plasticity: strengthen active synapses proportional to input
        self.adaptation += 0.004 * input_signal
        self.adaptation = np.clip(self.adaptation, 0.0, 2.0)

        # 7. Decay short-term plasticity variables each time step
        self.facilitation *= 0.95
        self.depression   *= 0.90
        # Ensure they stay non-negative
        self.facilitation = max(self.facilitation, 0.0)
        self.depression   = max(self.depression,   0.0)

        # 8. Record mean activity for heat-map visualization
        self.history.append(float(np.mean(self.neural_state)))

        # 9. Output: combine mean activity, adaptation, and spike contribution
        output = np.mean(self.neural_state) * (1.0 + self.adaptation) + 0.015 * spike_count
        print(f"[Living Node] Adaptation: {self.adaptation:.2f} | "
              f"Mean act: {np.mean(self.neural_state):.3f} | "
              f"Spikes: {spike_count} -> Output: {output:.2f}")
        return output

    # ------------------------------------------------------------------
    # Simple learning: boost adaptation when similar thought patterns repeat
    # ------------------------------------------------------------------
    def apply_learning_bonus(self, thought: str):
        """Increase a small bonus to adaptation based on keyword similarity."""
        # Normalise thought: lower case, split on non-alphanumeric
        words = [w for w in thought.lower().replace('-', ' ').split() if w]
        if not words:
            return
        # Use the first word as a simple pattern key (could be expanded)
        pattern = words[0]
        self.thought_boost[pattern] += 1
        # Each occurrence adds a tiny boost (capped)
        bonus = min(0.02 * self.thought_boost[pattern], 0.1)  # max +0.1
        self.adaptation = min(self.adaptation + bonus, 2.0)

# ----------------------------------------------------------------------
# Mock FinalSpark / wet-ware API wrapper
# ----------------------------------------------------------------------
class FinalSparkAPI:
    """
    Very simple mock of a wet-ware / FinalSpark style API.
    It receives spike data and returns:
        - metabolic_cost: a fake energy cost proportional to total spikes
        - processed_result: a dict with summary statistics
    """
    @staticmethod
    def process(spike_events, mean_activity_history):
        """
        spike_events: list of (time_step, neuron_index)
        mean_activity_history: list of mean activity per step (for context)
        Returns a tuple (metabolic_cost: float, processed_result: dict)
        """
        total_spikes = len(spike_events)
        # Fake metabolic cost: each spike costs 0.001 units, plus a baseline
        metabolic_cost = 0.05 + 0.001 * total_spikes

        # Processed result: some simple stats
        if spike_events:
            steps, neurons = zip(*spike_events)
            avg_step   = np.mean(steps)
            avg_neuron = np.mean(neurons)
            spike_rate = total_spikes / max(len(mean_activity_history), 1)
        else:
            avg_step = avg_neuron = spike_rate = 0.0

        processed_result = {
            "total_spikes": total_spikes,
            "avg_spike_time_step": round(float(avg_step), 2),
            "avg_spike_neuron_index": round(float(avg_neuron), 2),
            "spikes_per_step": round(float(spike_rate), 3),
            "note": "Mock FinalSpark response - replace with real API later."
        }
        return metabolic_cost, processed_result

# ----------------------------------------------------------------------
# Symbiotic AI - context-aware suggestions using full conversation history + node state.
# ----------------------------------------------------------------------
def symbiotic_ai(user_input: str, node: LivingNode, conversation_history: list, patch: NeuralPatch = None) -> str:
    """
    Returns a two-part response:
      1. Imaginative spark (metaphor/insight)
      2. Concrete next step (actionable suggestion)
    Uses the full conversation history, node state, and optional patch state to tailor the suggestion.
    """
    # Seed randomness with node output for reproducibility within a session
    seed_val = int((node.adaptation * 1e6) % 2**32)
    local_random = random.Random(seed_val)

    # ---- Creative spark (now context-aware) ----
    # Base sparks modified by node state
    base_sparks = [
        "A quiet spark flickers in the substrate, hinting at a hidden pattern.",
        "Your thought ripples through the wet-ware like a stone in a still pond.",
        "The neural lattice hums, resonating with a frequency just beyond awareness.",
        "A faint glow appears at the edge of the cortical map - something new is forming.",
        "Echoes of your intent bounce between dendrites, seeking a path forward.",
        "The bio-digital interface tingles, as if a synapse just fired for the first time.",
        "Patterns emerge from the noise, like constellations appearing in a cloudy sky.",
        "A whisper of understanding passes through the glial network.",
        "The membrane potential shifts, preparing for an action potential of insight.",
        "Silence breaks, and a micro-burst of activity cascades across the network."
    ]

    # Modify spark selection based on node state
    if node.adaptation > 1.5:
        # High adaptation: confident, flowing
        spark_modifiers = [
            "The pathways are well-worn; insight flows with ease.",
            "Your neural symphony plays in harmonious resonance.",
            "Established circuits amplify your intent into clear output."
        ]
    elif node.adaptation < 0.5:
        # Low adaptation: plastic, exploratory
        spark_modifiers = [
            "The substrate is highly plastic, ready for novel connections.",
            "Like fresh clay, the neural mass awaits your imprint.",
            "Low adaptation signals a window for radical new patterns."
        ]
    else:
        spark_modifiers = [
            "Neural activity balances between stability and change.",
            "The system is in a receptive state for new input."
        ]

    if node.facilitation > 0.5:
        spark_modifiers.append("Recent activity has facilitated synaptic transmission.")
    if node.depression > 0.5:
        spark_modifiers.append("Some pathways are depressed, suggesting a need for rest or novelty.")

    burst_msg = ""
    if patch and patch.burst_counter > 3:
        burst_msg = "A burst of rapid thoughts has primed the network."

    # Combine base spark with modifiers
    base_spark = local_random.choice(base_sparks)
    modifier = local_random.choice(spark_modifiers) if spark_modifiers else ""
    spark = f"{base_spark} {modifier}".strip()
    if burst_msg:
        spark += " " + burst_msg

    # ---- Context-aware practical next step ----
    # Analyze conversation history for topics
    recent_text = " ".join(conversation_history[-10:]).lower() if conversation_history else ""
    words = set(recent_text.split())

    # Determine action based on multiple factors
    if len(user_input.strip()) == 0:
        action = "Try holding Enter for a few seconds to build a sustained signal."
    elif node.adaptation < 0.3:
        action = "Your neural substrate is plastic-explore radically different angles. What's the opposite viewpoint?"
    elif node.adaptation > 1.5:
        action = "You're in a high-adaptation state-refine and document what's working well."
    else:
        # Medium adaptation: context-dependent suggestions
        if any(w in words for w in ["design", "plan", "idea", "concept", "sketch", "draft", "build", "create"]):
            action = "Sketch a quick diagram of the core idea, labeling at least three key components and their relationships."
        elif any(w in words for w in ["problem", "issue", "challenge", "obstacle", "barrier"]):
            action = "Break the problem into smaller sub-problems and tackle the easiest one first."
        elif any(w in words for w in ["share", "talk", "explain", "teach", "write", "present", "communicate"]):
            action = "Write a one-sentence summary you could share with a colleague, then note a concrete follow-up."
        elif any(w in words for w in ["learn", "study", "research", "investigate", "explore"]):
            action = "Spend 5 minutes gathering one concrete piece of information related to this thought."
        else:
            # Default based on node output (we don't have it here, but we can estimate)
            # We'll use a heuristic: if recent thoughts are short, suggest elaboration; if long, suggest distillation
            avg_len = np.mean([len(t) for t in conversation_history[-3:]]) if len(conversation_history) >= 3 else 0
            if avg_len < 20:
                action = "Elaborate on this thought: what details, implications, or connections come to mind?"
            else:
                action = "Distill this thought to its essence: what is the single most important point?"

    # Add facilitation/depression influence
    if node.facilitation > 0.7:
        action += " Consider reinforcing this line of thought with repetition or practice."
    if node.depression > 0.7:
        action += " Some pathways are fatigued-consider switching to a different topic or taking a short break."

    return f"{spark}\n[Idea] {action}"

# ----------------------------------------------------------------------
# Brainstorming session - extended creative AI interaction
# ----------------------------------------------------------------------
def brainstorm_session(node: LivingNode, conversation_history: list, patch: NeuralPatch):
    """
    Trigger a longer, creative session using the AI.
    Generates multiple related ideas based on current state and history.
    """
    print("\n" + "="*50)
    print("*** BRAINSTORMING SESSION ***")
    print("="*50)
    print(f"Current State: Adaptation={node.adaptation:.2f}, "
          f"Facilitation={node.facilitation:.2f}, Depression={node.depression:.2f}")
    if conversation_history:
        print(f"Recent thoughts: {', '.join(repr(t) for t in conversation_history[-3:])}")
    print("-"*50)

    # Generate 4 different brainstorming responses
    for i in range(4):
        # Vary the context slightly for each response
        # We'll adjust the random seed based on iteration
        seed_val = int(((node.adaptation + i*0.1) * 1e6) % 2**32)
        local_random = random.Random(seed_val)

        # Pick a base spark and modify it
        base_sparks = [
            "A constellation of ideas forms in the neural fog.",
            "Like neurons firing in succession, a chain of thoughts emerges.",
            "The cortical map lights up with overlapping activation patterns.",
            "Synaptic potentials summate, reaching threshold for insight.",
            "Glial cells modulate the extracellular environment, shaping signal flow.",
            "Dendritic branches explore new territory in search of connection.",
            "Axonal transmission carries the signal forward with varying fidelity.",
            "Myelin sheaths insulate pathways, increasing conduction velocity.",
            "Nodes of Ranvier facilitate saltatory conduction of understanding.",
            "Neurotransmitters bind to receptors, triggering postsynaptic potentials."
        ]
        base_spark = local_random.choice(base_sparks)

        # Contextual modifier based on node state
        modifiers = []
        if node.adaptation > 1.0:
            modifiers.append("High adaptation allows rapid chaining of related ideas.")
        elif node.adaptation < 0.5:
            modifiers.append("Low plasticity encourages novel, divergent thinking.")

        if node.facilitation > 0.4:
            modifiers.append("Recent facilitation strengthens associations between concepts.")
        if node.depression > 0.4:
            modifiers.append("Some depression suggests avoiding over-used pathways.")

        modifier = local_random.choice(modifiers) if modifiers else ""
        spark = f"{base_spark} {modifier}".strip()

        # Generate a contextualized idea
        # We'll create a pseudo-thought based on history and state
        if conversation_history:
            last_thought = conversation_history[-1]
            # Simple variation: add a perspective shift
            if local_random.random() < 0.5:
                pseudo_thought = f"What if we considered the opposite of '{last_thought}'?"
            else:
                pseudo_thought = f"How does '{last_thought}' connect to broader systems?"
        else:
            pseudo_thought = "Explore the foundational assumptions behind your current focus."

        # Get action suggestion using our enhanced AI (but we'll simulate the call)
        # For simplicity, we'll generate a basic action here
        if len(pseudo_thought.strip()) == 0:
            action = "Hold Enter to build a signal for deep contemplation."
        elif node.adaptation < 0.3:
            action = "Radically reframe the problem-what assumptions are you taking for granted?"
        elif node.adaptation > 1.5:
            action = "Document the current line of thinking and explore its practical applications."
        else:
            if any(w in pseudo_thought.lower() for w in ["design", "plan", "idea"]):
                action = "Sketch how this idea could be implemented in a simple prototype."
            elif any(w in pseudo_thought.lower() for w in ["problem", "challenge"]):
                action = "Identify the smallest experiment that could test this idea."
            else:
                action = "Take one concrete step forward: write down the next actionable item."

        if node.facilitation > 0.6:
            action += " Repeat this thought process to strengthen the neural pathway."
        if node.depression > 0.6:
            action += " Consider switching modalities (e.g., sketch instead of write) to refresh engagement."

        print(f"[Spark {i+1}] {spark}")
        print(f"[Idea {i+1}] {action}")
        print()

    print("="*50)
    print("End of brainstorming session.")
    print("="*50 + "\n")

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
            thought_boost = data.get('thought_boost', {})
            print(f"[Memory] Loaded previous state. Adaptation: {adaptation:.3f}")
            return neural_state, adaptation, thought_boost
        except Exception as e:
            print(f"[Memory] Failed to load: {e}")
    return None, 0.0, {}

def save_memory(neural_state, adaptation, thought_boost):
    try:
        data = {
            'neural_state': [float(x) for x in neural_state],
            'adaptation': float(adaptation),
            'thought_boost': dict(thought_boost)
        }
        with open(MEMORY_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[Memory] Saved state. Adaptation: {adaptation:.3f}")
    except Exception as e:
        print(f"[Memory] Failed to save: {e}")

# ----------------------------------------------------------------------
# Plotting - heat-map and raster
# ----------------------------------------------------------------------
def plot_results(history, spike_events, save_heatmap=HEATMAP_FILE, save_raster=RASTER_FILE):
    """Save both a heat-map of mean activity and a raster plot of spikes."""
    # ----- Heat-map (mean activity over time) -----
    if not history:
        print("[Plot] No history to visualize.")
    else:
        data = np.array(history)                     # shape: (steps,)
        # Create a 20-row representation for a raster-like view
        matrix = np.tile(data, (20, 1)) + np.random.normal(0, 0.01, (20, len(data)))

        plt.figure(figsize=(10, 4))
        plt.imshow(matrix.T, aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Mean Neural Activity')
        plt.title('Tarpluto v0.8 - Neural Activity Over Time (Heat-map)')
        plt.xlabel('Time Step')
        plt.ylabel('Neuron Layer (synthetic)')
        plt.tight_layout()
        os.makedirs('demos', exist_ok=True)
        plt.savefig(save_heatmap, dpi=150)
        plt.close()
        print(f"[Plot] Saved heat-map visualization to {save_heatmap}")

    # ----- Raster plot (spike timing) -----
    if not spike_events:
        print("[Plot] No spike events to rasterize.")
    else:
        steps, neurons = zip(*spike_events)  # unzip
        plt.figure(figsize=(10, 4))
        plt.scatter(steps, neurons, s=10, c='black', marker='.', linewidths=0)
        plt.title('Tarpluto v0.8 - Spike Raster Plot')
        plt.xlabel('Time Step')
        plt.ylabel('Neuron Index')
        plt.ylim(-0.5, 999.5)   # because we have 1000 neurons (0-999)
        plt.tight_layout()
        os.makedirs('demos', exist_ok=True)
        plt.savefig(save_raster, dpi=150)
        plt.close()
        print(f"[Plot] Saved raster visualization to {save_raster}")

# ----------------------------------------------------------------------
# Main interaction loop with command handling
# ----------------------------------------------------------------------
def main():
    print("\n=== Tarpluto Simulation v0.8 ===\n")
    print("Commands: status, save, fatigue, help, quit, api, brainstorm")
    print("Enter a thought/idea, or hold Enter to build sustained signal.\n")

    neural_state, adaptation, thought_boost = load_memory()
    patch = NeuralPatch()
    node = LivingNode(neural_state=neural_state, adaptation=adaptation)
    # restore learned thought boosts
    node.thought_boost = defaultdict(int, thought_boost)
    recent_thoughts = deque(maxlen=3)   # for immediate context (kept for compatibility)
    conversation_history = []           # full history of thoughts (limited to 50)

    try:
        while True:
            start = time.time()
            user_input = input("> ").rstrip('\n')
            think_time = time.time() - start

            cmd = user_input.strip().lower()
            if cmd == 'quit':
                break
            elif cmd == 'help':
                print("Available commands:")
                print("  status  - show current adaptation, burst counter, recent activity, and plasticity")
                print("  save    - force save memory & both visualizations now")
                print("  fatigue - temporarily halve adaptation to simulate mental fatigue")
                print("  help    - show this help")
                print("  quit    - exit and save")
                print("  api     - send current state to mock FinalSpark API and show response")
                print("  brainstorm - trigger an extended creative AI session")
                print("Or type any thought/idea to process it.")
                continue
            elif cmd == 'status':
                recent = np.mean(node.history[-10:]) if len(node.history) >= 10 else (
                    np.mean(node.history) if node.history else 0.0)
                print(f"[Status] Adaptation: {node.adaptation:.3f} | "
                      f"Burst: {patch.burst_counter} | "
                      f"Facilitation: {node.facilitation:.3f} | "
                      f"Depression: {node.depression:.3f} | "
                      f"Recent mean act: {recent:.3f} | "
                      f"History steps: {len(node.history)} | "
                      f"Recent thoughts: {list(recent_thoughts)}")
                continue
            elif cmd == 'save':
                plot_results(node.history, node.spike_events)
                save_memory(node.neural_state, node.adaptation, node.thought_boost)
                continue
            elif cmd == 'fatigue':
                # temporarily halve adaptation; effect lasts until next input
                node.adaptation *= 0.5
                print(f"[Fatigue] Adaptation temporarily halved to {node.adaptation:.3f}")
                continue
            elif cmd == 'api':
                # Query the mock FinalSpark API with current spike events and history
                if not node.spike_events and not node.history:
                    print("[API] No data to send yet. Think a few thoughts first.")
                else:
                    cost, result = FinalSparkAPI.process(node.spike_events, node.history)
                    print(f"[FinalSpark API] Metabolic cost: {cost:.4f}")
                    print(f"[FinalSpark API] Processed result:")
                    for k, v in result.items():
                        print(f"    {k}: {v}")
                continue
            elif cmd == 'brainstorm':
                # Trigger brainstorming session
                brainstorm_session(node, conversation_history, patch)
                continue

            # Normal thought processing
            signal = patch.read_thought(user_input)
            result = node.process(signal)

            # Apply simple learning bonus for repeated similar thoughts
            node.apply_learning_bonus(user_input)

            # Update recent thoughts for context-aware AI (ignore pure commands)
            if cmd not in ('status', 'save', 'fatigue', 'help', 'api', 'brainstorm'):
                recent_thoughts.append(user_input)
                conversation_history.append(user_input)
                # Limit conversation history to last 50 entries to prevent unbounded growth
                if len(conversation_history) > 50:
                    conversation_history.pop(0)

            advice = symbiotic_ai(user_input, node, conversation_history, patch)
            print(f"[Symbiotic AI]\n{advice}\n")
            print(f"[Output] Final result: {result:.3f}\n")
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nInterrupted!")
    finally:
        print("\nFinalizing...")
        plot_results(node.history, node.spike_events)
        save_memory(node.neural_state, node.adaptation, node.thought_boost)
        print("Remember: You are the symbiosis. Keep building.\n")

if __name__ == "__main__":
    main()