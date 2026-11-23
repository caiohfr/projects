import base64
from html import escape
from pathlib import Path

import streamlit as st

from src.vde_app.components.shared import render_inline_image
from src.vde_app.home_context import VDE_SETUP_STATE_KEY, resolve_home_context


def page_methodology():
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

    # ======= SCIENTIFIC METHODOLOGY =======
    st.header("🧠 Scientific Methodology")
    # use absolute path provided by the project owner (simple and direct)
    ABS_DIR = Path(__file__).resolve().parent          # pasta do arquivo atual
    abs_path = ABS_DIR.parent / 'data' / 'images' / 'vehicle-longitudinal-dynamics.png'
    render_inline_image(abs_path, width=1000)

    st.subheader("1. Core Concept")

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
    | **MVP2** | Regression & Scenario Analysis + Insert Components Maps & Gather more vehicle info  | ⚙️ In Progress |
    | **MVP3** | Transmission Losses + BEV/PHEV extensions | 🔜 Planned |
    | **MVP4** | ML Application using Scikit Learning to determine specific components efficiency | 🔜 Planned | 
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

def _home_styles() -> None:
    st.markdown(
        """
        <style>
        .ed-hero {padding: 1.4rem 1.55rem; border: 1px solid #d8e5ec; border-radius: 18px;
          background: linear-gradient(135deg, #f4fbf8 0%, #eef6fb 100%); margin-bottom: 1rem;}
        .ed-eyebrow {font-size:.76rem; letter-spacing:.12em; font-weight:700; color:#16765f;}
        .ed-hero h1 {font-size:2.05rem; margin:.25rem 0 .4rem; color:#173342;}
        .ed-hero p {font-size:1.02rem; color:#516875; margin:0; max-width:760px;}
        .ed-section {font-size:1.1rem; font-weight:700; color:#173342; margin:.9rem 0 .45rem;}
        .ed-context {min-height:132px; padding:1rem; border:1px solid #dbe6ec; border-radius:14px; background:#fff;}
        .ed-context .kind {font-size:.7rem; letter-spacing:.09em; color:#16765f; font-weight:700; text-transform:uppercase;}
        .ed-context h3 {font-size:1rem; margin:.35rem 0 .2rem; color:#173342;}
        .ed-context p {font-size:.82rem; color:#60727c; margin:.15rem 0;}
        .ed-empty {padding:1rem; border:1px dashed #b9cad3; border-radius:14px; color:#60727c;}
        div[data-testid="stPageLink"] a {border-radius:10px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _context_card(kind: str, title: str, detail: str, source: str) -> None:
    st.markdown(
        f"""<div class="ed-context"><div class="kind">{escape(kind)}</div><h3>{escape(title)}</h3>
        <p>{escape(detail)}</p><p><strong>Source:</strong> {escape(source)}</p></div>""",
        unsafe_allow_html=True,
    )


def _page_home_v03():
    _home_styles()
    context = resolve_home_context(st.session_state)

    st.markdown(
        """<section class="ed-hero"><div class="ed-eyebrow">ECODRIVE / ENGINEERING WORKSPACE</div>
        <h1>Engineering Workspace</h1>
        <p>Keep the same vehicle context across demand setup, load-case exploration,
        powertrain analysis and persisted comparisons.</p></section>""",
        unsafe_allow_html=True,
    )

    action_left, action_right, _ = st.columns([1.25, 1.25, 4.5])
    if context.has_active_analysis:
        with action_left:
            if st.button("Continue analysis", type="primary", width="stretch", key="home_continue"):
                st.switch_page("pages/VDE_Setup.py")
        with action_right:
            with st.popover("Start another proposal", width="stretch"):
                st.warning("This clears only the temporary VDE Setup session. Persisted records are not deleted.")
                confirmed = st.checkbox("Discard the current temporary setup", key="home_confirm_new_proposal")
                if st.button("Discard and start", disabled=not confirmed, key="home_start_another"):
                    st.session_state.pop(VDE_SETUP_STATE_KEY, None)
                    st.switch_page("pages/VDE_Setup.py")
    else:
        with action_left:
            if st.button("New VDE proposal", type="primary", width="stretch", key="home_new_proposal"):
                st.switch_page("pages/VDE_Setup.py")

    for warning in context.warnings:
        st.warning(warning)

    st.markdown('<div class="ed-section">Active Context</div>', unsafe_allow_html=True)
    baseline_col, proposal_col, fuel_col = st.columns(3)
    with baseline_col:
        if context.baseline:
            _context_card("Baseline", context.baseline.title, context.baseline.detail, context.baseline.source)
        else:
            st.markdown('<div class="ed-empty"><strong>Baseline</strong><br>No recoverable VDE selected.</div>', unsafe_allow_html=True)
    with proposal_col:
        if context.proposal:
            _context_card("Proposal", context.proposal.title, context.proposal.detail, context.proposal.source)
        else:
            st.markdown('<div class="ed-empty"><strong>Proposal</strong><br>No configured temporary proposal.</div>', unsafe_allow_html=True)
    with fuel_col:
        if context.fuelcons:
            _context_card("FuelCons reference", context.fuelcons.title, context.fuelcons.detail, context.fuelcons.source)
        else:
            st.markdown('<div class="ed-empty"><strong>FuelCons reference</strong><br>No persisted L0 reference selected.</div>', unsafe_allow_html=True)

    st.markdown('<div class="ed-section">Start an engineering task</div>', unsafe_allow_html=True)
    first, second, third = st.columns(3)
    with first:
        st.page_link("pages/VDE_Setup.py", label="VDE Setup")
        st.caption("Resolve a baseline and configure proposals by engineering domain.")
        st.page_link("pages/Comparison_Report.py", label="Comparison Report")
        st.caption("Compare persisted FuelCons records or inspect persisted physical VDEs.")
    with second:
        if context.baseline:
            if st.button("Roadload & Load Case Lab", width="stretch", key="home_open_lab"):
                st.session_state["roadload_load_case_vde_id"] = int(context.baseline.vde_id)
                st.switch_page("pages/Roadload_Load_Case_Lab.py")
            st.caption(f"Open persisted baseline VDE #{context.baseline.vde_id}; Lab remains in-memory and does not save.")
        else:
            st.button("Roadload & Load Case Lab", disabled=True, width="stretch", key="home_open_lab_disabled")
            st.caption("Select a persisted VDE baseline first. Temporary preview handoff is not supported yet.")
        st.page_link("pages/Powertrain_Scenario.py", label="Powertrain Scenario")
        st.caption("Select a saved FuelCons reference before running the L0 workflow.")
    with third:
        st.page_link("pages/Database_Management.py", label="Database Management")
        st.caption("Review persisted engineering records and their provenance.")

    st.markdown('<div class="ed-section">Recent Analysis</div>', unsafe_allow_html=True)
    if not context.recent:
        st.info("No recoverable recent analysis in this session. Start from a persisted baseline or FuelCons record.")
    else:
        for item in context.recent:
            left, right = st.columns([5, 1.4], vertical_alignment="center")
            left.markdown(f"**{item.title}**  \n{item.detail} · {item.source}")
            right.page_link(item.destination, label="Open")

    st.markdown('<div class="ed-section">Reference & Tools</div>', unsafe_allow_html=True)
    docs_col, tools_col = st.columns(2)
    with docs_col:
        st.page_link("pages/Methodology_Models.py", label="Methodology & Models")
        st.caption("Browse the existing scientific explanation independently from the working context.")
    with tools_col:
        st.page_link("pages/Legacy_Engineering_Tools.py", label="Engineering Tools / Legacy")
        st.caption("Advanced, diagnostic and compatibility surfaces outside the primary workflow.")


def _hero_image_uri() -> str:
    image_path = Path(__file__).resolve().parents[1] / "data" / "images" / "ecodrive_home_hero_v02.jpg"
    try:
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    except OSError:
        return ""
    return f"data:image/jpeg;base64,{encoded}"


def _home_styles_v02(hero_uri: str) -> None:
    hero_background = (
        f"linear-gradient(90deg,rgba(5,34,55,.98) 0%,rgba(8,48,74,.9) 43%,rgba(8,48,74,.2) 72%),"
        f"url('{hero_uri}')"
        if hero_uri
        else "linear-gradient(112deg,#082940 0%,#0a3c60 54%,#136b87 100%)"
    )
    st.markdown(
        f"""
        <style>
        .ed-home-hero {{
          min-height:220px; border-radius:9px; padding:1.65rem 2rem; margin-bottom:.55rem;
          display:flex; flex-direction:column; justify-content:center; color:#fff;
          background-image:{hero_background}; background-size:cover; background-position:center;
          border:1px solid rgba(255,255,255,.12); box-shadow:0 7px 24px rgba(8,41,64,.16);
        }}
        .ed-home-brand {{font-size:1.45rem; font-weight:760; margin-bottom:.75rem;}}
        .ed-home-eyebrow {{font-size:.7rem; letter-spacing:.13em; font-weight:750; color:#aee8df;}}
        .ed-home-hero h1 {{font-size:2.18rem; line-height:1.05; max-width:570px; margin:.28rem 0 .55rem; color:#fff;}}
        .ed-home-hero p {{font-size:.92rem; max-width:590px; margin:0; color:#e2edf3;}}
        .ed-section-title {{font-size:1.02rem; font-weight:760; color:var(--ed-ink); margin:1rem 0 .45rem;}}
        .ed-section-note {{font-size:.76rem; color:var(--ed-muted); margin-top:-.3rem; margin-bottom:.7rem;}}
        .ed-task-icon {{
          width:42px; height:42px; display:grid; place-items:center; border-radius:9px;
          background:#eaf3ff; color:var(--ed-blue); font-size:1.25rem; margin-bottom:.55rem;
        }}
        .ed-task-title {{font-weight:760; color:var(--ed-ink); font-size:.92rem; margin-bottom:.25rem;}}
        .ed-task-copy {{color:var(--ed-muted); font-size:.76rem; line-height:1.4; min-height:2.2rem;}}
        .ed-context {{min-height:126px; padding:.9rem; border:1px solid var(--ed-border); border-radius:8px; background:#fff;}}
        .ed-context .kind {{font-size:.66rem; letter-spacing:.09em; color:var(--ed-green); font-weight:750; text-transform:uppercase;}}
        .ed-context h3 {{font-size:.9rem; margin:.3rem 0 .18rem; color:var(--ed-ink);}}
        .ed-context p {{font-size:.76rem; color:var(--ed-muted); margin:.12rem 0;}}
        .ed-empty {{min-height:126px; padding:.9rem; border:1px dashed #b9cad3; border-radius:8px; color:var(--ed-muted); background:#fff;}}
        .ed-flow {{display:flex; align-items:center; justify-content:space-between; gap:.55rem; padding:.55rem .2rem;}}
        .ed-flow-step {{text-align:center; color:var(--ed-ink); font-size:.72rem; flex:1;}}
        .ed-flow-icon {{font-size:1.25rem; color:var(--ed-blue); display:block; margin-bottom:.2rem;}}
        .ed-flow-arrow {{color:#91abc0; font-size:1.1rem;}}
        .ed-capability {{font-size:.77rem; color:var(--ed-ink); margin:.34rem 0;}}
        .ed-capability span {{color:var(--ed-green); font-weight:800; margin-right:.35rem;}}
        [class*="st-key-home_"] button {{border-radius:7px;}}
        @media (max-width:1024px) {{
          .ed-home-hero {{min-height:205px; padding:1.35rem; background-position:58% center;}}
          .ed-home-hero h1 {{font-size:1.75rem; max-width:480px;}}
          .ed-flow {{flex-wrap:wrap;}}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _task_card(icon: str, title: str, copy: str, destination: str) -> None:
    with st.container(border=True):
        st.markdown(
            f'<div class="ed-task-icon">{escape(icon)}</div><div class="ed-task-title">{escape(title)}</div>'
            f'<div class="ed-task-copy">{escape(copy)}</div>',
            unsafe_allow_html=True,
        )
        st.page_link(destination, label=title, icon=":material/arrow_forward:")


def page_home():
    """Render the v0.2 visual identity with the validated v0.3 session flows."""

    _home_styles_v02(_hero_image_uri())
    context = resolve_home_context(st.session_state)

    st.markdown(
        """<section class="ed-home-hero">
        <div class="ed-home-brand">❧ EcoDrive</div>
        <div class="ed-home-eyebrow">VEHICLE DATA · ENGINEERING INSIGHT</div>
        <h1>Turn vehicle data into engineering insight.</h1>
        <p>Analyze demand, compare evidence and explore scenarios in one transparent engineering workspace.</p>
        </section>""",
        unsafe_allow_html=True,
    )

    action_left, action_right, action_space = st.columns([1.35, 1.55, 4.4])
    if context.has_active_analysis:
        with action_left:
            if st.button("Continue analysis", type="primary", width="stretch", key="home_continue"):
                st.switch_page("pages/VDE_Setup.py")
        with action_right:
            with st.popover("Start another proposal", width="stretch"):
                st.warning("This clears only the temporary VDE Setup session. Persisted records are not deleted.")
                confirmed = st.checkbox("Discard the current temporary setup", key="home_confirm_new_proposal")
                if st.button("Discard and start", disabled=not confirmed, key="home_start_another"):
                    st.session_state.pop(VDE_SETUP_STATE_KEY, None)
                    st.switch_page("pages/VDE_Setup.py")
    else:
        with action_left:
            if st.button("New VDE proposal", type="primary", width="stretch", key="home_new_proposal"):
                st.switch_page("pages/VDE_Setup.py")

    for warning in context.warnings:
        st.warning(warning)

    st.markdown('<div class="ed-section-title">Start an engineering task</div>', unsafe_allow_html=True)
    st.markdown('<div class="ed-section-note">Open a real EcoDrive workflow while keeping the current engineering context.</div>', unsafe_allow_html=True)
    comparison, setup, powertrain, database = st.columns(4)
    with comparison:
        _task_card("▥", "Comparison Report", "Benchmark persisted results and inspect program gaps.", "pages/Comparison_Report.py")
    with setup:
        _task_card("▣", "VDE Setup", "Build and resolve vehicle-demand proposals.", "pages/VDE_Setup.py")
    with powertrain:
        _task_card("⌘", "Powertrain Scenario", "Evaluate demand-linked fuel and electrical consumption.", "pages/Powertrain_Scenario.py")
    with database:
        _task_card("▤", "Database Management", "Review engineering records and provenance.", "pages/Database_Management.py")

    st.markdown('<div class="ed-section-title">Roadload &amp; Load Case Lab</div>', unsafe_allow_html=True)
    with st.container(border=True):
        lab_copy, lab_action = st.columns([4.8, 1.3], vertical_alignment="center")
        lab_copy.markdown(
            "Explore a temporary load case from a validated persisted baseline. "
            "The Lab remains in memory and does not save the scenario."
        )
        with lab_action:
            if context.baseline:
                if st.button("Open Lab", type="primary", width="stretch", key="home_open_lab"):
                    st.session_state["roadload_load_case_vde_id"] = int(context.baseline.vde_id)
                    st.switch_page("pages/Roadload_Load_Case_Lab.py")
            else:
                st.button("Select baseline first", disabled=True, width="stretch", key="home_open_lab_disabled")

    st.markdown('<div class="ed-section-title">Active Context</div>', unsafe_allow_html=True)
    baseline_col, proposal_col, fuel_col = st.columns(3)
    with baseline_col:
        if context.baseline:
            _context_card("Baseline", context.baseline.title, context.baseline.detail, context.baseline.source)
        else:
            st.markdown('<div class="ed-empty"><strong>Baseline</strong><br>No recoverable VDE selected.</div>', unsafe_allow_html=True)
    with proposal_col:
        if context.proposal:
            _context_card("Proposal", context.proposal.title, context.proposal.detail, context.proposal.source)
        else:
            st.markdown('<div class="ed-empty"><strong>Proposal</strong><br>No configured temporary proposal.</div>', unsafe_allow_html=True)
    with fuel_col:
        if context.fuelcons:
            _context_card("FuelCons reference", context.fuelcons.title, context.fuelcons.detail, context.fuelcons.source)
        else:
            st.markdown('<div class="ed-empty"><strong>FuelCons reference</strong><br>No persisted L0 reference selected.</div>', unsafe_allow_html=True)

    flow_col, capability_col = st.columns([1.7, 1])
    with flow_col:
        st.markdown('<div class="ed-section-title">How EcoDrive works</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(
                """<div class="ed-flow">
                <div class="ed-flow-step"><span class="ed-flow-icon">▤</span>Data &amp;<br>Evidence</div><div class="ed-flow-arrow">→</div>
                <div class="ed-flow-step"><span class="ed-flow-icon">▣</span>Vehicle<br>Demand</div><div class="ed-flow-arrow">→</div>
                <div class="ed-flow-step"><span class="ed-flow-icon">⌘</span>Powertrain<br>Energy</div><div class="ed-flow-arrow">→</div>
                <div class="ed-flow-step"><span class="ed-flow-icon">▥</span>Comparison<br>&amp; Decision</div>
                </div>""",
                unsafe_allow_html=True,
            )
    with capability_col:
        st.markdown('<div class="ed-section-title">Key capabilities</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(
                """<div class="ed-capability"><span>●</span>Regulatory-aligned methodology</div>
                <div class="ed-capability"><span>●</span>Scenario-based analysis</div>
                <div class="ed-capability"><span>●</span>Transparent provenance</div>
                <div class="ed-capability"><span>●</span>Component-level evaluation</div>""",
                unsafe_allow_html=True,
            )

    st.markdown('<div class="ed-section-title">Recent Analysis</div>', unsafe_allow_html=True)
    if not context.recent:
        st.info("No recoverable recent analysis in this session. Start from a persisted baseline or FuelCons record.")
    else:
        for item in context.recent:
            left, right = st.columns([5, 1.2], vertical_alignment="center")
            left.markdown(f"**{item.title}**  \n{item.detail} · {item.source}")
            right.page_link(item.destination, label="Open")

    docs_col, tools_col = st.columns(2)
    with docs_col:
        with st.container(border=True):
            st.markdown("**Methodology & Documentation**  \nUnderstand the engineering foundation behind EcoDrive.")
            st.page_link("pages/Methodology_Models.py", label="Methodology & Models")
    with tools_col:
        with st.container(border=True):
            st.markdown("**Engineering Tools**  \nOpen advanced, diagnostic and compatibility workflows.")
            st.page_link("pages/Legacy_Engineering_Tools.py", label="Engineering Tools / Legacy")


if __name__ == "__main__":
    page_home()
