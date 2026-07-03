# Tarpluto - Core Simulation Loop v0.1
# Neural Patch (input) -> Living Node (wetware sim) -> Output

import numpy as np
import time
import matplotlib.pyplot as plt

class NeuralPatch:
    def __init__(self):
        self.signal_strength = 0.0

    def read_thought(self, intent="default"):
        """Mock brain signal from thought/intent"""
        self.signal_strength = np.random.uniform(0.6, 1.0)
        print(f"[Neural Patch] Reading thought: '{intent}' | Strength: {self.signal_strength:.2f}")
        return self.signal_strength

class LivingNode:
    def __init__(self):
        self.neural_state = np.zeros(100)  # Simple simulated neuron population
        self.adaptation = 0.0

    def process(self, input_signal):
        """Wetware-style parallel processing simulation"""
        # Simulate biological integration + plasticity
        self.neural_state = np.clip(self.neural_state + input_signal * np.random.randn(100) * 0.3, 0, 1)
        self.adaptation += 0.05  # Learning/plasticity

        output = np.mean(self.neural_state) * (1 + self.adaptation)
        print(f"[Living Node] Processing... Adaptation: {self.adaptation:.2f} | Output strength: {output:.2f}")
        return output

# Main loop
if __name__ == "__main__":
    patch = NeuralPatch()
    node = LivingNode()

    print("=== Tarpluto Simulation Started ===\n")

    for i in range(5):  # Simulate 5 thought cycles
        intent = input("Enter a thought/intent (or press Enter for random): ") or f"idea_{i}"
        signal = patch.read_thought(intent)
        result = node.process(signal)

        print(f"[Output] Augmented result: {result:.2f}\n")
        time.sleep(0.8)

    print("Simulation complete. Extend this with real BCI + wetware APIs.")