"""Presentation-only styles shared by the Sprint 15D surfaces."""

from __future__ import annotations

import streamlit as st

from src.vde_app.visual_theme import BLUE, BORDER, GREEN, INK, MUTED, NAVY, WORKSPACE_BG


def inject_sprint15d_styles() -> None:
    """Reuse the approved shell tokens without replacing native controls."""

    st.markdown(
        f"""
        <style>
        :root {{
          --s15d-navy:{NAVY}; --s15d-blue:{BLUE}; --s15d-green:{GREEN};
          --s15d-bg:{WORKSPACE_BG}; --s15d-border:{BORDER};
          --s15d-ink:{INK}; --s15d-muted:{MUTED};
        }}
        .s15d-kicker {{
          color:var(--s15d-blue); font-size:.67rem; font-weight:780;
          letter-spacing:.11em; text-transform:uppercase; margin-bottom:.15rem;
        }}
        .s15d-lead {{
          color:var(--s15d-muted); font-size:.82rem; line-height:1.45;
          margin:-.25rem 0 .75rem; max-width:920px;
        }}
        .s15d-status-strip {{
          display:grid; grid-template-columns:repeat(3,minmax(0,1fr));
          margin:.2rem 0 .85rem; border:1px solid var(--s15d-border);
          border-radius:8px; background:#fff; overflow:hidden;
          box-shadow:0 3px 12px rgba(11,41,70,.025);
        }}
        .s15d-status-item {{padding:.62rem .78rem; border-left:1px solid var(--s15d-border);}}
        .s15d-status-item:first-child {{border-left:0;}}
        .s15d-status-item span {{
          display:block; color:var(--s15d-muted); font-size:.62rem;
          letter-spacing:.055em; text-transform:uppercase;
        }}
        .s15d-status-item strong {{
          display:block; margin-top:.12rem; color:var(--s15d-ink);
          font-size:.78rem; overflow-wrap:anywhere;
        }}
        .s15d-status-item.is-green strong {{color:#0d7351;}}
        .s15d-section-label {{
          margin:.8rem 0 .35rem; color:var(--s15d-ink);
          font-size:.86rem; font-weight:760;
        }}
        .s15d-note {{
          padding:.58rem .72rem; margin:.35rem 0 .7rem;
          border-left:3px solid var(--s15d-blue); border-radius:0 6px 6px 0;
          color:#36566f; background:#eef5fa; font-size:.72rem; line-height:1.42;
        }}
        .s15d-note.is-green {{border-left-color:var(--s15d-green); background:#f0faf6;}}
        .s15d-note.is-warn {{border-left-color:#c68a20; background:#fff8e8; color:#765617;}}
        .s15d-card-grid {{
          display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.65rem;
          margin:.45rem 0 .8rem;
        }}
        .s15d-card {{
          min-height:8rem; padding:.78rem; border:1px solid var(--s15d-border);
          border-radius:8px; background:#fff;
        }}
        .s15d-card-icon {{
          display:grid; place-items:center; width:2rem; height:2rem;
          border-radius:7px; color:var(--s15d-blue); background:#eaf4fa;
          font-size:.83rem; font-weight:800;
        }}
        .s15d-card strong {{display:block; margin:.55rem 0 .25rem; color:var(--s15d-ink); font-size:.82rem;}}
        .s15d-card p {{margin:0; color:var(--s15d-muted); font-size:.69rem; line-height:1.38;}}
        .s15d-card em {{display:block; margin-top:.5rem; color:var(--s15d-blue); font-size:.65rem; font-style:normal; font-weight:760;}}
        .s15d-card.is-disabled {{background:#f8fafc; opacity:.78;}}
        .s15d-card.is-disabled em {{color:var(--s15d-muted);}}
        .s15d-article-header {{
          padding:.85rem .95rem; margin-bottom:.65rem; border:1px solid var(--s15d-border);
          border-top:3px solid var(--s15d-blue); border-radius:8px; background:#fff;
        }}
        .s15d-article-header h2 {{margin:.18rem 0 .25rem; color:var(--s15d-navy); font-size:1.35rem;}}
        .s15d-article-header p {{margin:0; color:var(--s15d-muted); font-size:.76rem; line-height:1.45;}}
        .s15d-badge {{
          display:inline-flex; padding:.15rem .48rem; border-radius:999px;
          color:#0d7351; background:#e8f7f0; font-size:.62rem; font-weight:760;
        }}
        div[data-testid="stTabs"] button[role="tab"] {{font-size:.76rem; font-weight:690;}}
        div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{background:var(--s15d-blue);}}
        [data-testid="stDataFrame"], [data-testid="stDataEditor"] {{
          border:1px solid var(--s15d-border); border-radius:7px; overflow:hidden;
        }}
        @media (max-width:1100px) {{
          .s15d-status-strip {{grid-template-columns:1fr;}}
          .s15d-status-item {{border-left:0; border-top:1px solid var(--s15d-border);}}
          .s15d-status-item:first-child {{border-top:0;}}
          .s15d-card-grid {{grid-template-columns:repeat(2,minmax(0,1fr));}}
        }}
        @media (max-width:700px) {{.s15d-card-grid {{grid-template-columns:1fr;}}}}
        </style>
        """,
        unsafe_allow_html=True,
    )


__all__ = ["inject_sprint15d_styles"]
