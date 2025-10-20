import streamlit as st

def page_home():
    st.title("🏠 EcoDrive Analyzer – Scientific Home Page")
    st.markdown("---")
    
    # ======= OVERVIEW =======
    st.header("🚗 Overview")
    st.markdown("""
**EcoDrive Analyzer** is a scientific platform for benchmarking and visualizing **Vehicle Demanded Energy (VDE)** —  
the minimum mechanical energy required for a vehicle to follow a driving cycle under standardized road-load conditions.

It enables engineers and researchers to:
- Quantify energy demand using coastdown coefficients (A/B/C).
- Compare vehicles under **EPA**, **WLTP**, and **PROCONVE** frameworks.
- Evaluate technology impacts (mass, aerodynamics, tires, hybridization).
- Explore real-world efficiency scenarios.
    """)

    st.markdown("---")
    # ======= PURPOSE =======
    st.header("🎯 Purpose")
    st.markdown("""
The goal is to build a **transparent, physics-based, and reproducible** framework for energy analysis and fuel-consumption correlation,  
aligned with public regulations (**SAE, EPA, UNECE, ISO**).

EcoDrive Analyzer bridges the gap between **regulatory energy metrics (VDE_NET)**  
and **applied engineering insights (VDE_TOTAL)** through an open scientific approach.
    """)

    st.markdown("---")
    # ======= SCIENTIFIC METHODOLOGY =======
    st.header("🧠 Scientific Methodology")

    st.subheader("1. Core Concept")
    #st.image(r'C:\Users\CaioHenriqueFerreira\Downloads\From Git\projects\EcoDrive-Analyst\data\images\longitudinal_dynamics.png',width= 1000)
    st.latex(r"VDE = \int (F_{road}(v) + m_{test} a(t)) \, v(t) \, dt")
    st.markdown("""
Where:
- \(F_{road}(v)\): resistive forces acting on the vehicle.
- \(m_{test}\): test mass (ETW / WLTP Test Mass).
- \(v(t), a(t)\): velocity and acceleration from the cycle trace.
    """)

    st.subheader("2. Road-Load Components")
    st.latex(r"F_{road}(v) = F_{roll} + F_{aero} + F_{parasitic} = A + Bv + Cv^2")
    st.markdown("""
Each term in the **road-load equation** represents a physical phenomenon:

| Term | Component | Typical Source | Units |
|:--:|:--|:--|:--:|
| **A** | Rolling + Parasitic losses | Tire deformation, bearings, drivetrain drag | N |
| **Bv** | Speed-proportional losses | Transmission, bearings, low-Reynolds drag | N/kph |
| **Cv²** | Aerodynamic drag | Air resistance (depends on Cd × Af) | N/kph² |

This total force \(F_{road}(v)\) defines how much **tractive effort** is required at the tire–road interface.
    """)

    st.subheader("3. Tractive Energy and VDE Relationship")
    st.latex(r"P_{tractive}(t) = (F_{road}(v) + m_{test} a(t)) \cdot v(t)")
    st.markdown("""
At each moment in the driving cycle, the **tractive power** \(P_{tractive}\) represents the power that must be delivered  
by the powertrain to overcome resistance and accelerate the vehicle.

By integrating this over time, we obtain:

\[
E_{tractive} = \int P_{tractive}(t)\,dt
\]

The **Vehicle Demanded Energy (VDE)** is then defined as this tractive energy, normalized by distance:

\[
VDE = \frac{E_{tractive}}{d_{cycle}} \;\; [MJ/km]
\]

Thus:
- **VDE_NET** represents the *ideal tractive energy* (without drivetrain losses).  
- **VDE_TOTAL** includes additional **transmission and accessory losses**, bringing it closer to real-world consumption.
    """)

    st.markdown("---")
    # ======= FRAMEWORK =======
    st.header("🔬 Analytical Framework")
    st.markdown("""
The analysis follows a transparent, reproducible data pipeline:
    """)

    st.graphviz_chart("""
    digraph {
        rankdir=LR;
        node [shape=box, style=rounded, fontsize=11];
        InputData [label="Input Data"];
        RoadLoad [label="Road-Load Coefficients (A/B/C)"];
        Mass [label="Test Mass"];
        Cycle [label="Cycle Trace v(t)"];
        Resist [label="Resistive Forces"];
        Inertia [label="Inertial Forces"];
        Power [label="Instantaneous Power"];
        Integration [label="Integration Over Time"];
        VDENet [label="VDE_NET [MJ/km]"];
        Losses [label="Add Transmission Losses?"];
        VDETotal [label="VDE_TOTAL [MJ/km]"];
        Output [label="Graphs & Reports"];

        InputData -> RoadLoad;
        InputData -> Mass;
        InputData -> Cycle;
        RoadLoad -> Resist;
        Mass -> Inertia;
        Cycle -> Resist;
        Cycle -> Inertia;
        Resist -> Power;
        Inertia -> Power;
        Power -> Integration;
        Integration -> VDENet;
        VDENet -> Losses;
        Losses -> VDETotal;
        VDETotal -> Output;
    }
    """)

    st.markdown("---")
    # ======= HOW TO USE =======
    st.header("🧩 How to Use the Dashboard")
    st.markdown("""
1. **Baseline Picker** – Select an existing vehicle snapshot from the database.  
   It pre-fills coastdown coefficients, mass, and cycle data.

2. **Compute & Save** – Calculates **VDE_NET** using physical inputs and saves it to the database.

3. **Parameters Card** – View and edit parameters like A/B/C, mass, or electrification type.  
   Use it to simulate changes in tires, aerodynamics, or weight.

4. **Regression Card** – Runs automatic regressions between **VDE** and **fuel consumption**,  
   showing how well energy demand correlates with measured efficiency.

5. **Comparison & Graphs** – Visualize effects of:
   - ΔMass (lightweighting)  
   - ΔCxAf (aerodynamics)  
   - ΔB (rolling resistance)  
   - Electrification level (ICE → MHEV → PHEV → BEV)
    """)

    st.markdown("---")
    # ======= DATA SOURCES =======
    st.header("📊 Data Sources")
    st.markdown("""
EcoDrive Analyzer uses **public and regulatory datasets** only:
- **EPA 40 CFR 1066** – U.S. vehicle testing procedures.  
- **UNECE GTR 15 (WLTP)** – Worldwide harmonized light vehicles test procedure.  
- **INMETRO / PBEV** – Brazilian fuel economy datasets.  
- **SAE J1263 / J2263** – Road-load and coastdown measurement standards.  
- **ICCT / SAE Technical Papers** – Energy and consumption correlation references.
    """)

    st.markdown("---")
    # ======= EQUATIONS =======
    st.header("🧮 Core Equations")

    st.table({
        "Symbol": ["F_road", "A, B, C", "m_test", "v(t)", "a(t)", "VDE_NET", "VDE_TOTAL"],
        "Meaning": [
            "Total resistive force",
            "Coastdown coefficients",
            "Test mass",
            "Speed trace",
            "Acceleration",
            "Energy per distance (no transmission losses)",
            "Energy per distance (with transmission losses)"
        ],
        "Units": ["N", "N, N/kph, N/kph²", "kg", "kph", "m/s²", "MJ/km", "MJ/km"]
    })

    st.markdown("---")
    # ======= ROADMAP =======
    st.header("🧭 Project Roadmap")

    st.markdown("""
| Phase | Focus | Status |
|:--|:--|:--:|
| **MVP0** | Core physics model (EPA/WLTP) | ✅ Completed |
| **MVP1** | Database integration + Streamlit dashboard | ✅ Near complete |
| **MVP2** | Regression and scenario analysis | ⚙️ In progress |
| **MVP3** | Transmission losses + BEV/PHEV modeling | 🔜 Planned |
    """)

    st.markdown("---")
    # ======= TRANSPARENCY =======
    st.header("📜 Transparency Principles")
    st.markdown("""
1. **Reproducibility:** all computations are deterministic and open.  
2. **Normative alignment:** consistent with SAE, ISO, EPA, UNECE.  
3. **Scientific clarity:** no black-box models.  
4. **Comparability:** harmonized metrics across regions.  
5. **Traceability:** every calculation linked to the database snapshot.
    """)

    st.markdown("---")
    # ======= CREDITS =======
    st.header("💬 Credits")
    st.markdown("""
Developed by **Caio H. F. Rocha**  
as part of the **CS50 Final Project** and ongoing research in automotive energy benchmarking.  
Built with **Python**, **Streamlit**, and **SQLite**, using open-access regulatory data.
    """)

if __name__ == "__main__":
    page_home()
