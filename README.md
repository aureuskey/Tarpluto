# Tarpluto

**Bio-Digital Symbiosis Infrastructure: Thought-Speed Intelligence Without Surgery**

## Overview
Tarpluto is an open‑source simulation of a non‑invasive Neural Patch coupled with a Living Node (wetware processor) that enables thought‑speed human‑AI symbiosis. The project simulates the core loop: user think‑time → neural signal → wet‑ware processing → symbiotic AI feedback → visualization and persistence.

## Current Capabilities (v0.8)
- **Neural Patch**: Converts think‑time and Enter‑hold duration into a controllable signal strength (0‑1) with burst detection.
- **Living Node**: 1,000‑neuron wetware simulation featuring:
  - Refractory period & dynamic spike threshold
  - Short‑term synaptic plasticity (facilitation & depression)
  - Spike raster recording & heat‑map activity tracking
  - Simple learning mechanism: repeated thoughts boost adaptation
- **Symbiotic AI**: Context‑aware suggestion engine that uses:
  - Full conversation history
  - Live node state (adaptation, facilitation, depression)
  - Neural Patch burst counter
  - Generates imaginative sparks + concrete next‑step ideas
- **Brainstorm Command**: Trigger an extended creative session that outputs multiple related idea sparks based on current neural state.
- **Mock FinalSpark API**: Simulated wet‑ware wrapper returning metabolic cost and processed spike statistics.
- **Persistence**: Automatic saving/loading of neural state (`memory.json`) and visualizations.
- **Visualization**: Generates:
  - `demos/simulation_heatmap.png` – mean neural activity over time
  - `demos/simulation_raster.png` – spike timing raster plot
- **Interactive CLI**: Commands `status`, `save`, `fatigue`, `help`, `quit`, `api`, `brainstorm`.

## How to Run
1. **Clone the repository** (if you haven’t already):
   ```bash
   git clone https://github.com/yourusername/Tarpluto.git
   cd Tarpluto
   ```
2. **Set up the environment** (Windows):
   ```bash
   .\setup.bat
   ```
   This creates a virtual environment, installs dependencies (`numpy`, `matplotlib`), and creates the needed folders.
3. **Activate the virtual environment**:
   ```bash
   .\venv\Scripts\activate
   ```
4. **Launch the simulation**:
   ```bash
   python sim\core_loop.py
   ```
5. **Interact**:
   - Type any thought or idea at the `>` prompt.
   - Use commands: `status`, `save`, `fatigue`, `brainstorm`, `api`, `help`, `quit`.
   - Visualizations are saved to the `/demos` folder; neural state persists in `memory.json`.

## Screenshots / Outputs
After running the simulation, check the `/demos` folder for generated visualizations:
- **Heat‑map** – `simulation_heatmap.png` (average neural activity over time)
- **Raster Plot** – `simulation_raster.png` (spike timing across 1,000 neurons)

*(Place actual screenshots here when available.)*

## Roadmap
See the detailed 30‑day insane plan: [docs/roadmap.md](docs/roadmap.md)

## Contributing & Feedback
We welcome collaborators, feedback, and ideas! Whether you’re interested in:
- Improving the wet‑ware neural model
- Integrating real EEG/OpenBCI hardware
- Refining the symbiotic AI with LLMs
- Expanding the visualization suite
- Documentation or outreach

Please open an issue or submit a pull request. Let’s build the future of human‑AI symbiosis together.

---

*Let's build.*