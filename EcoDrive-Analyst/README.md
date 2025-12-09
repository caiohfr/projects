# ⚡ EcoDrive Analyzer

## 👨‍🔬 CS50 Explaning Video

**Eco Drive Analyzer**
- YouTube: [Watch on YouTube](https://youtu.be/yQ4lP8M8Cw0)


---

**Vehicle Demanded Energy (VDE) Benchmarking Tool**

A scientific and transparent platform for vehicle energy analysis, regulatory comparison, and efficiency visualization.
Developed as part of the **CS50 Final Project** by *Caio H. F. Rocha* (2025).

---

## 🚗 Overview

**EcoDrive Analyzer** quantifies and visualizes the **Vehicle Demanded Energy (VDE)** —
the minimum mechanical energy required for a vehicle to follow a driving cycle under standardized road-load conditions.

It provides a **transparent, physics-based** framework aligned with **EPA, WLTP, and PROCONVE** methodologies (WLTP Lack of Data, PROCONVE (BR) To be implemented).

---
## 🌱 Motivation

**EcoDrive Analyzer** was created as part of my CS50 journey — a way to connect **engineering precision with human clarity**.
It reflects my aim to build tools that make efficiency measurable, transparent, and grounded in physics.

Through this project, I learned to **value clarity over speed**, to turn theory into something reproducible, and to find meaning in making science visible.

> “It’s not about doing more. It’s about doing with clarity and impact.”

---

## 🧠 Core Concept

\[
VDE = \int (F_{road}(v) + m_{test} a(t)) \, v(t) \, dt
\]

- \(F_{road}(v) = A + Bv + Cv^2\)
- \(m_{test}\): equivalent test mass (ETW / WLTP)
- \(v(t), a(t)\): speed and acceleration profiles from cycle traces

The **VDE** is then normalized by the total cycle distance → expressed in **MJ/km**.

---

## ⚙️ Road-Load Components

| Symbol | Description | Main Source | Unit |
|:--:|:--|:--|:--:|
| **A** | Rolling + Parasitic Resistance | Tire deformation, bearings, drivetrain drag | N |
| **Bv** | Speed-proportional Losses | Transmission, bearings | N/kph |
| **Cv²** | Aerodynamic Drag | Air resistance (Cd × Af) | N/kph² |

---

## 🔋 Tractive and Demanded Energy

\[
P_{tractive}(t) = (F_{road}(v) + m_{test} a(t)) \cdot v(t)
\]

\[
VDE_{NET} = \frac{1}{d_{cycle}} \int P_{tractive}(t)\,dt
\]

- **VDE_NET** → ideal tractive energy (transmission in neutral).
- **VDE_TOTAL** → includes drivetrain losses for real-world correlation.

---

## 🧩 Features

- **📥 VDE Setup** – load driving cycles and parameters (A/B/C, mass).
- **⚙️ PWT Fuel Energy** – compute Vehicle Demanded Energy (VDE_NET).
- ***🧮 Regression Card*** – correlate VDE with fuel consumption based on EPA Tested Data (WLTP missing data for now).
- **💾 Local Database (SQLite)** – stores every snapshot and calculation.
- **📊 Operating Points / Report (TO DO)** – visualize PWT operating points based on Transmission & Engine or Electric Motor.
- **🧠 Scenario Analysis** – test efficiency improvements (mass, aero, tires). Comparison Report to be improved bur it can be done by VDE Setup.

---

## 🧪 Scientific Methodology

The analysis pipeline follows a transparent and reproducible process:

1. **Input Data:** coastdown coefficients, mass, and cycle trace.
2. **Compute VDE_NET:** road-load + inertia + integration over time.
3. **Add Transmission Losses (optional):** derive VDE_TOTAL.
4. **Regression Analysis:** correlate VDE with measured fuel consumption.
5. **Visualization:** energy breakdown, deltas, and correlations.

---

## 🧬 Data Sources

- 🇺🇸 **EPA 40 CFR 1066** – U.S. vehicle test procedure.
- 🇪🇺 **UNECE GTR 15 / WLTP** – Worldwide harmonized light vehicles test.
- 🇧🇷 **INMETRO / PBEV** – Brazilian fuel economy dataset.
- **SAE J1263 / J2263** – Road-load and coastdown measurement standards.
- **SAE 2020-01-1064 / ICCT 2014** – Transmission losses and correlation studies.

---

## 🖥️ Tech Stack

| Layer | Technology |
|:--|:--|
| Frontend | Streamlit |
| Backend | Python 3.11 |
| Database | SQLite3 |
| Visualization | Plotly / Matplotlib |
| Data Handling | Pandas / NumPy |
| Reports | Markdown / PDF (via ReportLab) |

---

## 📊 Machine Learning Analysis (scikit-learn)

The project includes a small ML module built with scikit-learn to validate the physical models and explore predictive relationships in the EPA dataset.

A first experiment used only categorical attributes (make, category, powertrain, transmission) to predict coastdown coefficients A/B/C, reaching R² ≈ 0.70, showing strong structural patterns across vehicle classes.

A second model combined coastdown coefficients and basic vehicle parameters to predict certified fuel consumption, achieving R² ≈ 0.96 with a Random Forest.
This aligns with vehicle physics: road load → RLHP → FE.

These models complement the physics-based VDE workflow and help validate data consistency, estimate missing values, and accelerate early-stage benchmarking.

---

## 🧰 Folder Structure

```
EcoDrive-Analyzer/
│
├── app.py                      # Main Streamlit entry point
├── data/
│   └── db/eco_drive.db         # Local database (auto-generated)
│   └── cycles/                 # Cycles USed
│   └── vehicles/               # Vehicles Data
│   └── standards/              # Standards
│   └── images/              # Standards
|
│
├── pages/
│   ├── home_page.py            # Home with methodology and theory
│   ├── vde_setup.py            # Load cycle and setup parameters
│   ├── pwt_fuel_energy.py      # Compute VDE & energy KPIs
│   └── operating_points.py     # Regression & comparison dashboards
│
├── src/
│   └── vde_core/               # Core calculation modules
│       ├── db.py
│       ├── services.py
│       ├── regression.py
│       └── utils.py
│   └── vde_app/               # app modules
│       ├── components.py
│       ├── derivatives.py
│       ├── plots.py
│       └── state.py
|
├── EPA_xlsx_to_db.ipynb       # Pre Processing data from EPA & Manage initial DB
|
└── README.md
```

---

## 🧮 Mathematical Layers

| Layer | Description | Output |
|:--|:--|:--:|
| **VDE_NET** | Vehicle Demanded Energy (neutral) | MJ/km |
| **VDE_TOTAL** | Includes drivetrain losses | MJ/km |
| **ΔTech Scenarios** | Compare aero, mass, tires, hybridization | % Δ MJ/km |

---

## 🧭 Roadmap (MVPs)

| Stage | Goal | Status |
|:--|:--|:--:|
| **MVP0** | Core physical computation (EPA/WLTP cycles) | ✅ Complete |
| **MVP1** | Database + Streamlit UI | ✅ Finalizing |
| **MVP2** | ML Regression & Scenario Analysis + Insert Components Maps & Gather more vehicle info  | ⚙️ In Progress |
| **MVP3** | Transmission Losses + BEV/PHEV extensions | 🔜 Planned |
| **MVP4** | ML Application using Scikit Learning to determine specific components efficiency | 🔜 Planned |

---

## 📈 Example Output

```
Cycle: FTP-75
-------------------------------------
VDE_urban_NET   = 1.82 MJ/km
VDE_highway_NET = 1.15 MJ/km
VDE_comb_NET    = 1.56 MJ/km
```

---

## 🧪 Installation

```bash
# Clone repository (root repo contains the EcoDrive-Analyst folder)
git clone https://github.com/caiohfr/projects.git
cd projects/EcoDrive-Analyst

# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

---

## 🧩 Usage

1. Launch the Streamlit interface:
   ```bash
   streamlit run app.py
   ```
2. Open your browser (`http://localhost:8501`).
3. Workflow:
   - **📥 VDE Setup:** Create new vehicle profile
   - **⚙️ PWT Fuel Energy:** PWT Data, correlate Fuel Economy w/ vehicle Data
   - **📊 Comparison Report: (TO DO)** Need more Data visualization & Analysis tool to be implemented
   - **📊 Operating Points: (TO DO)** analyze specific points during cycle, need PWT specific maps. Future Application

---

## 🧾 Transparency & Reproducibility

- 100% based on **public, normative data and formulas**.
- No proprietary models or closed datasets.
- All results traceable to physical inputs and database entries.

---

## 👨‍🔬 Author

**Caio H. F. Rocha**
*Automotive Engineer MSc – Stellantis / CS50 Student*
- LinkedIn: [LinkedIn - Caio H. F. Rocha](https://www.linkedin.com/in/caio-henrique-ferreira-rocha-728011140/)
- GitHub: [GitHub - caiohfr](https://github.com/caiohfr)

---

## 🧠 License

This project is distributed under the **MIT License**.
You are free to use, modify, and distribute with proper credit.

---

## 🏁 Acknowledgments

- **Harvard CS50** – for the foundation in computational thinking.
- **SAE / ICCT / UNECE** – for the open regulatory frameworks.
- **Streamlit Team** – for simplifying scientific dashboards.

---

### 📚 Citation

If you use this project in research or academic work, please cite as:

> Rocha, C.H.F. (2025). *EcoDrive Analyzer – A Transparent Tool for Vehicle Demanded Energy Benchmarking*. CS50 Final Project.
