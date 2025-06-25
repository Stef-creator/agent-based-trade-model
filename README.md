# International Agent-Based Trade Model (Llerena & Lorentz Replication and Extensions)

This repository contains a Python-based implementation and extension of the international agent-based model (ABM) originally proposed by **Llerena & Lorentz (2004)**. The model simulates endogenous growth and trade across multiple economies, focusing on innovation, firm-level competition, exchange rate dynamics, and now — **trade tariffs and policy regimes**.

---

## Purpose

- Replicate the **baseline evolutionary ABM**.
- Introduce **FX dynamics** and **tariff regimes** to explore trade competitiveness.
- Quantitatively assess the **persistence of economic leadership**, **path-dependence**, and **regime-lock in**.
- Evaluate empirical outcomes across **1,000 simulations** with and without tariff intervention.

---

## Contents

- `sim_models.py`: All simulation model logic and analysis functions.
- `main_notebook.ipynb`: Full simulation setup, results, visualizations, and interpretations.
- `figures/`: Output plots and distributional summaries.
- `README.md`: This file.
- `requirements.txt`: List of Python dependencies.

---

## How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/Stef-creator/agent-based-trade-model.git
   cd agent-based-trade-model

2. Create and activate virtual environment
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate

3. Install dependencies
    ```bash
    pip install -r requirements.txt

4. Launch Jupyter Notebook
    jupyter notebook notebook.ipynb


---

## Key Features

- **Dynamic Exchange Rates**  
  Modeled based on export growth differential and random shocks.

- **Tariff Module**  
  Tariff-imposing economy penalizes import competitiveness via a user-defined tariff rate.

- **Leadership Tracking & Statistics**  
  Tracks:
  - Who leads initially
  - Whether they remain leader
  - Number of leadership transitions
  - Final GDP and export gaps

- **Visualization Dashboard**  
  Distribution plots for:
  - Final leader identity
  - Regime persistence
  - GDP & export divergence
  - Leadership churn

---

## Visualization Preview

Plots include:

- Final leader frequency (by economy)
- Persistent leadership share
- GDP and export dominance (boxplots)
- Number of leadership transitions (histogram)

---

## Future Extensions

- Add **entry and exit dynamics** for firms.
- Implement **endogenous tariff retaliation** strategies.
- Introduce **heterogeneous firm sizes**, **credit constraints**, or **sectoral shocks**.
- Calibrate model to real-world trade data (e.g., WTO, IMF).
- Visualize agent-level heterogeneity (e.g., productivity dispersion).

---

## Citation

If you use or adapt this work, please cite the original paper:

> Llerena, P. & Lorentz, A. (2004). *Cumulative Causation and Evolutionary Micro-Founded Technical Change.*

---

## Contact

Created by **Stefan Pilegaard Pedersen**  

---

## License

MIT License. See `LICENSE` file for details.
