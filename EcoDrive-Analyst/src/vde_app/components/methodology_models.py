"""Navigable, evidence-aligned Methodology & Models documentation."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Callable

import streamlit as st

from src.vde_app.components.shared import render_inline_image
from src.vde_app.components.sprint15d_style import inject_sprint15d_styles


_TOPICS = (
    "Vehicle Demanded Energy",
    "Roadload & Coastdown",
    "Powertrain L0",
    "Tires & Components",
    "Contracts / TOTAL vs NET",
    "Experimental Methods",
    "References & Provenance",
)


def _article_header(title: str, description: str, status: str) -> None:
    st.markdown(
        (
            '<div class="s15d-article-header">'
            f'<span class="s15d-badge">{escape(status)}</span>'
            f'<h2>{escape(title)}</h2><p>{escape(description)}</p>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def _vehicle_demand() -> None:
    _article_header(
        "Vehicle Demanded Energy",
        "Canonical wheel-side cycle integration from authoritative roadload, effective test mass and the cycle trace.",
        "Canonical method",
    )
    image_path = Path(__file__).resolve().parents[3] / "data" / "images" / "vehicle-longitudinal-dynamics.png"
    render_inline_image(image_path, width=1000)
    st.markdown("#### Core concept")
    st.latex(r"VDE = \frac{1}{d_{cycle}}\int_{0}^{T}\max(0, P_{tractive}(t))\,dt")
    st.latex(r"P_{tractive}(t) = [F_{road}(v(t)) + m_{test}a(t)]\,v(t)")
    st.markdown(
        "The Vehicle Demand layer answers what the vehicle requires at the wheels. "
        "It does not model how the powertrain supplies that demand. The canonical result "
        "integrates only positive tractive power for VDE while keeping roadload, inertial work, "
        "braking requirement and provenance at their resolved boundaries. In SI units, force is in N, "
        "speed in m/s, power in W, time in s and cycle distance in km, producing MJ/km after conversion."
    )
    st.markdown(
        '<div class="s15d-note is-green"><strong>Boundary:</strong> braking energy is a wheel-side requirement, not recovered regenerative energy. '
        'Negative tractive power is classified as braking demand and receives no regenerative-energy credit in VDE. '
        'A missing quantity remains unavailable; an explicit zero remains a real value.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("#### Inputs and outputs")
    st.table(
        {
            "Role": ["Inputs", "Outputs", "Traceability"],
            "Contract": [
                "Authoritative ABC, effective test mass, cycle v(t)/a(t)",
                "MJ/km summaries and time-resolved force/power profiles",
                "Roadload basis, mass source, model provenance and warnings",
            ],
        }
    )


def _roadload() -> None:
    _article_header(
        "Roadload & Coastdown",
        "Authoritative coastdown coefficients remain the reference boundary; known contributions explain but do not rewrite them.",
        "Canonical boundary",
    )
    st.latex(r"F_{road}(v) = A + Bv + Cv^2")
    st.table(
        {
            "Term": ["A", "Bv", "Cv²"],
            "Interpretation": [
                "Speed-independent part of the authoritative fit",
                "Speed-proportional part of the authoritative fit",
                "Speed-squared part of the authoritative fit",
            ],
            "Units": ["N", "N per km/h", "N per (km/h)²"],
        }
    )
    st.markdown(
        "The fitted coefficients are an authoritative roadload representation, not a guaranteed one-to-one "
        "physical decomposition. Known rolling and aero contributions are shown only when their own "
        "inputs are resolved; the remaining signed residual closes against the authoritative curve."
    )
    st.markdown(
        '<div class="s15d-note"><strong>Steady state:</strong> at fixed speed, acceleration is zero. A mass-only override does not fabricate a force delta when no active force contribution depends on mass.</div>',
        unsafe_allow_html=True,
    )


def _powertrain_l0() -> None:
    _article_header(
        "Powertrain L0",
        "A cycle-effective energy-balance layer that consumes resolved Vehicle Demand and explicit system assumptions.",
        "Current L0",
    )
    st.markdown("#### Interpretation flow")
    st.markdown("**Resolved Vehicle Demand → cycle-effective PSE assumptions → fuel, electric energy and CO₂ outputs**")
    st.markdown(
        "PSE is a cycle-effective system efficiency at the L0 boundary. It is not pure engine efficiency "
        "and it is not a hidden decomposition of engine, transmission, battery or controls. Physical "
        "configuration changes are distinguished from effects quantitatively represented at L0."
    )
    st.table(
        {
            "Runtime method": ["Manual / Imported", "Physics Simple", "Regression", "ML Prediction"],
            "Reading": [
                "Observed or explicitly supplied result",
                "Aggregate demand/efficiency estimate",
                "Dataset-dependent statistical estimate",
                "Artifact-dependent inference with explicit availability state",
            ],
        }
    )
    st.markdown(
        '<div class="s15d-note is-warn"><strong>Fidelity limit:</strong> L0 does not claim a time-resolved physical powertrain simulation or causal subsystem allocation.</div>',
        unsafe_allow_html=True,
    )


def _tires_components() -> None:
    _article_header(
        "Tires & Components",
        "Catalog evidence is applied only through implemented physical boundaries and explicit provenance.",
        "Evidence governed",
    )
    st.markdown(
        "Tire rolling resistance may contribute when an explicit RRC and effective load are available. "
        "Aero requires explicit CdA and resolved density. Transmission, brake, axle/hubs and parasitic "
        "records carry their own boundary, resolution method and provenance."
    )
    st.markdown("#### Evidence states")
    st.table(
        {
            "State": ["Canonical / measured", "Rule-estimated", "Configuration only", "Unavailable"],
            "Meaning": [
                "Directly supported by the selected record and boundary",
                "Estimated by an identified rule with provenance",
                "Physical metadata changed, but no quantitative effect is represented here",
                "Required evidence is absent; no zero or default is invented",
            ],
        }
    )
    st.markdown(
        '<div class="s15d-note"><strong>Authoritative identity:</strong> Known contribution + signed residual = authoritative roadload. The UI never forces known components to sum to 100%.</div>',
        unsafe_allow_html=True,
    )


def _total_net_contract() -> None:
    _article_header(
        "Contracts / TOTAL vs NET",
        "TOTAL and NET are explicit, distinct roadload boundaries with no fallback between them.",
        "Frozen contract",
    )
    st.markdown(
        "**TOTAL** is the authoritative coastdown boundary. **NET** exists only when the "
        "transmission-neutral loss is resolved, and is then calculated as TOTAL minus that "
        "resolved loss. Neither boundary silently substitutes for the other."
    )
    st.graphviz_chart(
        """
        digraph {
            rankdir=LR;
            node [shape=box, style=rounded, fontsize=11];
            Coast [label="Authoritative coastdown ABC"];
            Total [label="TOTAL boundary"];
            Neutral [label="Transmission-neutral loss resolved?"];
            Net [label="NET = TOTAL - resolved neutral loss"];
            Missing [label="NET unavailable"];
            Demand [label="Vehicle Demand profile / summary"];
            Coast -> Total;
            Total -> Demand;
            Total -> Neutral;
            Neutral -> Net [label="yes"];
            Neutral -> Missing [label="no"];
            Net -> Demand;
        }
        """
    )
    st.table(
        {
            "Boundary": ["TOTAL", "NET"],
            "Definition": [
                "Authoritative coastdown roadload and its cycle demand",
                "TOTAL minus resolved transmission-neutral loss, only when that boundary is available",
            ],
            "Fallback": ["Never reconstructed from NET", "Never substituted with TOTAL"],
        }
    )
    st.markdown(
        '<div class="s15d-note is-green">Historical rows that stored TOTAL in the former NET field are normalized only by the canonical data contract. Presentation code must not reinterpret fields independently.</div>',
        unsafe_allow_html=True,
    )


def _experimental() -> None:
    _article_header(
        "Experimental Methods",
        "Research and compatibility paths stay separate from canonical results and carry their own validation burden.",
        "Experimental / historical",
    )
    st.markdown(
        "The repository retains regression, ML inference, peer analysis, historical comparison tools and "
        "diagnostic paths. Availability depends on data and artifacts. A result is not upgraded to canonical "
        "because a renderer exists."
    )
    st.table(
        {
            "Capability": ["Regression", "ML Prediction", "Physics + ML residual", "Map-based simulation", "Component estimation"],
            "Current status": ["Runtime method", "Runtime when artifact/features are available", "Planned", "Planned", "Unavailable as a published workflow"],
        }
    )
    with st.expander("Historical roadmap context", expanded=False):
        st.markdown(
            "Earlier project material used MVP0–MVP4 labels. Those labels are retained as historical context, "
            "not as current release certification. Current capability status is defined by implemented routes, "
            "contracts, tests and per-result provenance."
        )


def _references() -> None:
    _article_header(
        "References & Provenance",
        "Engineering claims are interpreted with the selected record, source classification, model boundary and internal contract evidence.",
        "Traceable sources",
    )
    st.markdown("#### Regulatory and engineering references represented in project records")
    st.markdown(
        "- EPA 40 CFR 1066 and related U.S. test procedures\n"
        "- UNECE GTR 15 / WLTP\n"
        "- INMETRO / PBEV source material where present in records\n"
        "- SAE J1263 / J2263 coastdown references\n"
        "- ISO / SAE tire rolling-resistance test families"
    )
    st.markdown("#### Internal sources of truth")
    st.code(
        "src/vde_core/vde_net_total_contract.py\n"
        "src/vde_core/vehicle_demand/\n"
        "docs/architecture/vehicle_demand_architecture.md\n"
        "docs/sprints/PACKAGE_7G_VDE_NET_TOTAL_CONTRACT.md\n"
        "tests/test_vehicle_demand_*",
        language="text",
    )
    st.markdown(
        "Source availability and authority are evaluated per record. EcoDrive includes deterministic physics, "
        "engineering assumptions, statistical methods and optional ML artifacts; provenance must distinguish them."
    )
    st.markdown("---")
    st.markdown(
        "Developed by **Caio H. F. Rocha** as part of the CS50 Final Project and ongoing automotive-energy research."
    )


_RENDERERS: dict[str, Callable[[], None]] = {
    "Vehicle Demanded Energy": _vehicle_demand,
    "Roadload & Coastdown": _roadload,
    "Powertrain L0": _powertrain_l0,
    "Tires & Components": _tires_components,
    "Contracts / TOTAL vs NET": _total_net_contract,
    "Experimental Methods": _experimental,
    "References & Provenance": _references,
}


def render_methodology_models() -> None:
    inject_sprint15d_styles()
    st.markdown('<div class="s15d-kicker">Engineering knowledge</div>', unsafe_allow_html=True)
    st.title("Methodology & Models")
    st.markdown(
        '<div class="s15d-lead">Navigable documentation of the implemented physics, model boundaries, contracts and evidence. Historical material is labelled rather than presented as current capability.</div>',
        unsafe_allow_html=True,
    )
    index_column, article_column = st.columns([1, 3.15], gap="large")
    with index_column:
        st.markdown('<div class="s15d-section-label">Engineering knowledge</div>', unsafe_allow_html=True)
        topic = st.radio(
            "Methodology topic",
            _TOPICS,
            key="methodology_topic",
            label_visibility="collapsed",
        )
        st.caption("Select one topic. Only the active article is rendered.")
    with article_column:
        _RENDERERS[topic]()


__all__ = ["render_methodology_models"]
