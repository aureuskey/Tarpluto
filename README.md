# Tarpluto
**Bio-Digital Symbiosis: Think. Compute. Evolve.**

## Vision
To merge biological computing with human cognition, creating a seamless thought-speed interface that augments creativity, learning, and problem-solving without invasive hardware. Tarpluto simulates this symbiosis today, paving the way for tomorrow's neuro-augmented tools.

## Current Status (v0.9)
- Solo founder, laptop-stage prototype
- Core simulation of Neural Patch → Living Node → Symbiotic AI loop
- Persistent conversation memory and session reflection
- Interactive CLI with brainstorming and analytics commands

**Key Features:**
- Neural Patch: converts think-time and Enter-hold into a signal (0-1) with burst detection
- Living Node: 1,000-neuron wetware model with refractory dynamics, spike threshold, short-term plasticity (facilitation/depression), spike raster & heat-map visualization
- Symbiotic AI: context-aware suggestions using full conversation history, node state, and burst level
- Brainstorm command: generates 5 connected, high-quality ideas with actionable steps
- Reflect command: analyzes session stats (adaptation trend, burst level, top words) and offers cognitive insights
- Persistence: automatic save/load of neural state (`memory.json`) and conversation history
- Visualizations: heat-map and raster plots saved to `/demos/`
- Mock FinalSpark API: returns metabolic cost and spike statistics
- Simple learning: repeated thoughts boost adaptation

## How to Run
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Tarpluto.git
   cd Tarpluto
   ```
2. **Set up the environment** (Windows)
   ```bash
   .\setup.bat
   ```
   Creates a virtual environment, installs `numpy` and `matplotlib`, and ensures the `/demos/` folder exists.
3. **Activate the virtual environment**
   ```bash
   .\venv\Scripts\activate
   ```
4. **Launch the simulation**
   ```bash
   python sim\core_loop.py
   ```
5. **Interact**
   - Type any thought or idea at the `>` prompt.
   - Use commands: `status`, `save`, `fatigue`, `brainstorm`, `reflect`, `api`, `help`, `quit`.
   - Visualizations appear in `/demos/simulation_heatmap.png` and `/demos/simulation_raster.png`.
   - Neural state and conversation history persist in `memory.json`.

## Visualizations
After each run, check the `/demos/` folder:
- **`simulation_heatmap.png`** – mean neural activity over time (heat-map)
- **`simulation_raster.png`** – spike timing raster plot across 1,000 neurons

## Roadmap
See the detailed 30-day insane plan: [docs/roadmap.md](docs/roadmap.md)

## Philosophy
- **Ruthlessly cut bureaucracy** – If a meeting, doc, or permission isn’t moving the code forward, drop it.
- **Set insane deadlines** – Work expands to fill the time you give it—give it days, not months.
- **First principles only** – Question every assumption; build from physics of neurons and information theory upward.
- **Iterate in public** – Share progress, learn from feedback, and build in the open.

## Join the Journey
We welcome feedback, ideas, and collaborators. Whether you want to improve the wet-ware model, integrate real EEG/OpenBCI hardware, refine the symbiotic AI with LLMs, or help with documentation—your contribution matters.

👉 Follow updates and join the conversation on **X**: [@YourXHandle](https://twitter.com/YourXHandle)
👉 Open an issue or submit a pull request on GitHub.

Let's build the future of human-AI symbiosis together.