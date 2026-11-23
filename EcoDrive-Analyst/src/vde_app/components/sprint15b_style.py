from __future__ import annotations

from html import escape

import streamlit as st

from src.vde_app.visual_theme import BLUE, BORDER, GREEN, INK, MUTED, NAVY, WORKSPACE_BG


def inject_sprint15b_styles() -> None:
    """Add page-local Sprint 15B presentation without changing widget behavior."""
    st.markdown(
        f"""
        <style>
        .ed-workflow-hero {{
          display:flex; align-items:flex-start; justify-content:space-between; gap:1.5rem;
          padding:1.15rem 1.25rem; margin:0 0 1rem; border-radius:10px;
          color:#fff; background:linear-gradient(112deg,#082940 0%,{NAVY} 54%,#136b87 100%);
          box-shadow:0 3px 12px rgba(11,41,70,.08);
        }}
        .ed-workflow-kicker {{
          color:#86d7ff; font-size:.68rem; font-weight:800; letter-spacing:.12em;
          text-transform:uppercase; margin-bottom:.28rem;
        }}
        .ed-workflow-title {{font-size:1.48rem; line-height:1.2; font-weight:760;}}
        .ed-workflow-copy {{color:#d8e8f2; font-size:.88rem; line-height:1.45; margin-top:.35rem; max-width:760px;}}
        .ed-workflow-status {{
          flex:0 0 auto; border:1px solid rgba(255,255,255,.28); border-radius:999px;
          padding:.38rem .72rem; color:#e9f8f3; background:rgba(22,159,115,.2);
          font-size:.76rem; font-weight:750; white-space:nowrap;
        }}
        .ed-vde-flow-label {{
          color:{BLUE}; font-size:.68rem; font-weight:800; letter-spacing:.11em;
          text-transform:uppercase; margin:.25rem 0 .45rem;
        }}
        .ed-vde-matrix-header {{
          min-height:2.2rem; display:flex; align-items:center; padding:.38rem .42rem;
          border:1px solid {BORDER}; border-radius:6px; background:#eaf3fb;
          color:{NAVY}; font-size:.7rem; font-weight:760; line-height:1.15;
        }}
        .ed-lab-context {{
          display:grid; grid-template-columns:repeat(auto-fit,minmax(145px,1fr)); gap:.65rem;
          background:#fff; border:1px solid {BORDER}; border-radius:9px;
          padding:.8rem .9rem; margin:.25rem 0 1rem; box-shadow:0 3px 12px rgba(11,41,70,.035);
        }}
        .ed-lab-context-item {{min-width:0;}}
        .ed-lab-context-label {{
          color:{MUTED}; font-size:.66rem; font-weight:750; letter-spacing:.08em;
          text-transform:uppercase; margin-bottom:.18rem;
        }}
        .ed-lab-context-value {{color:{INK}; font-size:.88rem; font-weight:680; overflow-wrap:anywhere;}}
        .ed-result-banner {{
          display:flex; justify-content:space-between; gap:1rem; align-items:center;
          background:#fff; border:1px solid {BORDER}; border-left:4px solid {GREEN};
          border-radius:8px; padding:.7rem .85rem; margin:.2rem 0 1rem;
          color:{INK}; font-size:.84rem;
        }}
        .ed-result-banner strong {{color:{GREEN};}}
        .ed-empty-guidance {{
          background:{WORKSPACE_BG}; border:1px dashed {BORDER}; border-radius:8px;
          padding:.8rem .9rem; color:{MUTED}; font-size:.86rem; margin:.5rem 0;
        }}
        div[data-testid="stForm"] {{
          border-color:{BORDER}; border-radius:9px; background:rgba(255,255,255,.72);
        }}
        @media (max-width:1024px) {{
          .ed-workflow-hero {{flex-direction:column; gap:.8rem;}}
          .ed-workflow-status {{white-space:normal;}}
          .ed-lab-context {{grid-template-columns:repeat(2,minmax(0,1fr));}}
          .ed-vde-matrix-header {{font-size:.64rem;}}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_workflow_intro(*, kicker: str, title: str, description: str, status: str) -> None:
    st.markdown(
        (
            '<div class="ed-workflow-hero">'
            '<div>'
            f'<div class="ed-workflow-kicker">{escape(kicker)}</div>'
            f'<div class="ed-workflow-title">{escape(title)}</div>'
            f'<div class="ed-workflow-copy">{escape(description)}</div>'
            '</div>'
            f'<div class="ed-workflow-status">{escape(status)}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def render_lab_baseline_context(row: dict, baseline_id: int) -> None:
    identity = " ".join(
        str(row.get(key) or "").strip()
        for key in ("make", "model", "year")
        if row.get(key) not in (None, "")
    ).strip() or "Unnamed VDE"
    items = (
        ("Persisted source", f"VDE-{baseline_id:03d}"),
        ("Vehicle", identity),
        ("Legislation", row.get("legislation") or "Unavailable"),
        ("Cycle", row.get("cycle") or row.get("cycle_name") or "Canonical VDE cycle"),
        ("Scenario state", "Temporary / unsaved"),
    )
    cells = "".join(
        '<div class="ed-lab-context-item">'
        f'<div class="ed-lab-context-label">{escape(str(label))}</div>'
        f'<div class="ed-lab-context-value">{escape(str(value))}</div>'
        '</div>'
        for label, value in items
    )
    st.markdown(f'<div class="ed-lab-context">{cells}</div>', unsafe_allow_html=True)


def render_calculated_result_banner(baseline_id: int) -> None:
    st.markdown(
        (
            '<div class="ed-result-banner">'
            f'<div><strong>Calculated result</strong> · persisted baseline VDE-{baseline_id:03d}</div>'
            '<div>Input edits take effect only after Calculate load case.</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


__all__ = [
    "inject_sprint15b_styles",
    "render_calculated_result_banner",
    "render_lab_baseline_context",
    "render_workflow_intro",
]
