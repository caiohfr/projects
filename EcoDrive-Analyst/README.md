# 🚗 Vehicle Demanded Energy (VDE) Analyzer  

## 📑 Overview
The **Vehicle Demanded Energy (VDE) Analyzer** is a benchmarking tool implemented in **Python + Streamlit** to compute the minimum energy required for a vehicle to follow a regulatory driving cycle.  

It provides:
- **VDE_NET** → baseline metric (neutral coastdown, regulatory comparability).  
- **VDE_TOTAL** → extended metric (includes drivetrain losses, better correlation with fuel/energy consumption).  

The project follows a **modular, incremental development (CS50-style)** approach: starting from a minimal core calculation and evolving into a dashboard with comparisons and scenario analysis.

---

## ⚙️ What is VDE?
\[
VDE = \int (F_{road}(v) + m_{test} \cdot a(t)) \cdot v(t) \, dt
\]

- **Components:**
  - Road-load coefficients (f0,f1,f2 or A,B,C).  
  - Test mass/inertia (ETW/TWC or WLTP).  
  - Cycle trace \(v(t)\).  

- **Interpretation:**
  - **VDE_NET:** academic/regulatory metric.  
  - **VDE_TOTAL:** extended metric with drivetrain losses.  

---

## 📦 Libraries Used
- **[Streamlit](https://streamlit.io/):** interactive UI, multipage dashboard.  
- **[Pandas](https://pandas.pydata.org/):** data handling (cycles, presets, catalogs).  
- **[NumPy](https://numpy.org/):** numerical integration, gradients, vectorization.  
- **[Plotly](https://plotly.com/python/):** interactive plots (power vs time, sensitivity charts).  
- **[Pydantic](https://docs.pydantic.dev/):** (planned) input validation and type safety.  
- **[OpenPyXL](https://openpyxl.readthedocs.io/):** Excel export.  
- **[ReportLab](https://www.reportlab.com/):** (optional) PDF reports.  

---

## 🧱 Program Structure & Architecture

### High-level layers
- **UI Layer (Streamlit):**  
  Pages in `pages/`, helpers in `src/vde_app/`. Handles user input, state, and visualization.  
- **Core Layer (`vde_core`):**  
  Pure functions, models, and business logic (no Streamlit dependencies).  
- **Data Layer:**  
  Local CSV/JSON files for cycles, presets, and technologies (abstracted through loaders, extensible to SQLite/DuckDB).  

### Folder layout

### USAGE Instalation
git clone https://github.com/caiohfr/projects.git
cd projects/EcoDrive-Analyst
pip install -r requirements.txt
streamlit run app.py
