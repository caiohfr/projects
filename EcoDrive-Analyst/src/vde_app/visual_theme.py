from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st


NAVY = "#0a2b46"
SIDEBAR_NAVY = "#082940"
BLUE = "#1769a6"
GREEN = "#169f73"
WORKSPACE_BG = "#f4f7fb"
BORDER = "#dce5ef"
INK = "#0b2742"
MUTED = "#6d8298"


def inject_shell_styles() -> None:
    """Apply the approved v0.2 identity to the shared Streamlit shell."""
    st.markdown(
        f"""
        <style>
        :root {{
          --ed-navy:{NAVY}; --ed-sidebar:{SIDEBAR_NAVY}; --ed-blue:{BLUE};
          --ed-green:{GREEN}; --ed-bg:{WORKSPACE_BG}; --ed-border:{BORDER};
          --ed-ink:{INK}; --ed-muted:{MUTED};
          --ed-shadow:0 3px 12px rgba(11,41,70,.035);
        }}
        [data-testid="stAppViewContainer"], [data-testid="stMain"] {{background:var(--ed-bg);}}
        [data-testid="stMainBlockContainer"] {{
          max-width:1710px; padding-top:1.15rem; padding-left:2rem; padding-right:2rem;
        }}
        [data-testid="stSidebar"] {{
          background:linear-gradient(180deg,#082940 0%,#0a2f49 55%,#0b263c 100%);
          border-right:1px solid rgba(255,255,255,.08);
        }}
        [data-testid="stSidebarContent"] {{padding-top:.35rem;}}
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] summary {{color:#dce9f3;}}
        [data-testid="stSidebarNav"] span {{color:#eaf3f8;}}
        [data-testid="stSidebarNav"] a {{border-radius:6px; margin:2px 8px; min-height:38px;}}
        [data-testid="stSidebarNav"] a:hover {{background:rgba(23,105,166,.42);}}
        [data-testid="stSidebarNav"] a[aria-current="page"] {{
          background:#17659c; box-shadow:inset 3px 0 0 #58c1ff;
        }}
        [data-testid="stSidebarNav"] [data-testid="stSidebarNavSeparator"] {{
          color:#8eb1c8; letter-spacing:.1em; font-size:.67rem; text-transform:uppercase;
        }}
        .ed-brand {{
          display:flex; align-items:center; gap:.62rem; padding:.8rem .85rem .72rem;
          color:white; font-size:1.08rem; font-weight:750; letter-spacing:-.01em;
          border-bottom:1px solid rgba(255,255,255,.12); margin-bottom:.35rem;
        }}
        .ed-brand-mark {{
          width:28px; height:28px; display:grid; place-items:center; border-radius:8px;
          background:linear-gradient(145deg,#f5ffff,#9dd7d1); color:#0a4960; font-size:1rem;
        }}
        .ed-shell-meta {{
          padding:.65rem .85rem; color:#9eb7c8; font-size:.7rem;
          border-top:1px solid rgba(255,255,255,.1); margin-top:.45rem;
        }}
        .ed-topbar {{
          min-height:58px; display:flex; align-items:center; justify-content:space-between; gap:1rem;
          background:#fff; border:1px solid var(--ed-border); border-radius:8px;
          padding:.65rem 1rem; margin:0 0 1rem; box-shadow:var(--ed-shadow);
        }}
        .ed-topbar-kicker {{font-size:.68rem; color:var(--ed-blue); font-weight:750; letter-spacing:.1em;}}
        .ed-topbar-title {{font-size:1rem; color:var(--ed-ink); font-weight:720;}}
        .ed-topbar-db {{font-size:.75rem; color:var(--ed-muted); text-align:right;}}
        div[data-testid="stPageLink"] a {{border-radius:7px;}}
        @media (max-width:1024px) {{
          [data-testid="stMainBlockContainer"] {{padding-left:1rem; padding-right:1rem;}}
          .ed-topbar {{align-items:flex-start; flex-direction:column;}}
          .ed-topbar-db {{text-align:left;}}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    st.sidebar.markdown(
        '<div class="ed-brand"><span class="ed-brand-mark">&#10087;</span><span>EcoDrive</span></div>',
        unsafe_allow_html=True,
    )


def render_context_bar(page_title: str, db_path: str) -> None:
    db_name = Path(db_path).name if db_path else "No database selected"
    st.markdown(
        f"""<div class="ed-topbar">
        <div><div class="ed-topbar-kicker">ENGINEERING WORKSPACE</div>
        <div class="ed-topbar-title">{escape(page_title)}</div></div>
        <div class="ed-topbar-db">Runtime database<br><strong>{escape(db_name)}</strong></div>
        </div>""",
        unsafe_allow_html=True,
    )
