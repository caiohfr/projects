from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

from src.vde_app.components.shared import get_legislation_icon, render_inline_image, search_logo


_REPO_ROOT = Path(__file__).resolve().parents[3]
_IMAGES_DIR = _REPO_ROOT / "data" / "images"
_LOGOS_DIR = _IMAGES_DIR / "logos"


def inject_v22_styles() -> None:
    st.markdown(
        """
        <style>
        .v22-branding-empty {
            border: 1px dashed color-mix(in srgb, var(--text-color, #111827) 20%, transparent);
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
            color: color-mix(in srgb, var(--text-color, #111827) 72%, white 28%);
            background: color-mix(in srgb, var(--secondary-background-color, #f8fafc) 85%, transparent);
            font-size: 0.9rem;
        }
        .v22-branding-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: color-mix(in srgb, var(--text-color, #111827) 62%, white 38%);
            margin-bottom: 0.25rem;
        }
        .v22-branding-value {
            font-size: 1rem;
            font-weight: 600;
            line-height: 1.25;
            color: var(--text-color, #111827);
        }
        .v22-branding-meta {
            font-size: 0.86rem;
            color: color-mix(in srgb, var(--text-color, #111827) 70%, white 30%);
            margin-top: 0.2rem;
        }
        .v22-step-header {
            display: flex;
            gap: 0.9rem;
            align-items: flex-start;
            justify-content: space-between;
            border: 1px solid color-mix(in srgb, var(--text-color, #111827) 10%, transparent);
            border-radius: 8px;
            padding: 0.9rem 1rem;
            margin: 0.35rem 0 1rem 0;
            background: color-mix(in srgb, var(--secondary-background-color, #f8fafc) 72%, transparent);
        }
        .v22-step-header-main {
            display: flex;
            gap: 0.8rem;
            align-items: flex-start;
            min-width: 0;
        }
        .v22-step-index {
            min-width: 2rem;
            height: 2rem;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.95rem;
            font-weight: 700;
            background: color-mix(in srgb, var(--primary-color, #2563eb) 14%, transparent);
            color: var(--text-color, #111827);
        }
        .v22-step-title {
            font-size: 1.05rem;
            font-weight: 700;
            line-height: 1.25;
            color: var(--text-color, #111827);
        }
        .v22-step-caption {
            margin-top: 0.2rem;
            font-size: 0.9rem;
            line-height: 1.35;
            color: color-mix(in srgb, var(--text-color, #111827) 72%, white 28%);
        }
        .v22-step-status {
            font-size: 0.84rem;
            font-weight: 600;
            white-space: nowrap;
            color: color-mix(in srgb, var(--text-color, #111827) 78%, white 22%);
        }
        .v22-context-strip {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 0.6rem;
            margin: 0.55rem 0 0.95rem 0;
        }
        .v22-context-item {
            border: 1px solid color-mix(in srgb, var(--text-color, #111827) 10%, transparent);
            border-radius: 8px;
            padding: 0.65rem 0.75rem;
            background: color-mix(in srgb, var(--secondary-background-color, #f8fafc) 66%, transparent);
            min-height: 4.1rem;
        }
        .v22-context-label {
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: color-mix(in srgb, var(--text-color, #111827) 62%, white 38%);
            margin-bottom: 0.25rem;
        }
        .v22-context-value {
            font-size: 0.96rem;
            font-weight: 600;
            line-height: 1.3;
            color: var(--text-color, #111827);
            overflow-wrap: anywhere;
        }
        .v22-sidebar-meta {
            margin: -0.2rem 0 0.85rem 0.1rem;
            padding-left: 0.15rem;
        }
        .v22-sidebar-summary {
            font-size: 0.84rem;
            font-weight: 600;
            line-height: 1.25;
            color: var(--text-color, #111827);
        }
        .v22-sidebar-detail {
            font-size: 0.78rem;
            line-height: 1.3;
            color: color-mix(in srgb, var(--text-color, #111827) 68%, white 32%);
            margin-top: 0.15rem;
        }
        .v22-pager-note {
            font-size: 0.8rem;
            color: color-mix(in srgb, var(--text-color, #111827) 68%, white 32%);
            margin-top: 0.45rem;
        }
        .v22-panel {
            border: 1px solid color-mix(in srgb, var(--text-color, #111827) 10%, transparent);
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            background: color-mix(in srgb, var(--secondary-background-color, #f8fafc) 62%, transparent);
            margin: 0.35rem 0 0.9rem 0;
        }
        .v22-panel-title {
            font-size: 0.98rem;
            font-weight: 700;
            color: var(--text-color, #111827);
            margin-bottom: 0.2rem;
        }
        .v22-panel-caption {
            font-size: 0.86rem;
            line-height: 1.35;
            color: color-mix(in srgb, var(--text-color, #111827) 70%, white 30%);
            margin-bottom: 0.65rem;
        }
        .v22-summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.7rem;
            margin: 0.25rem 0 0.95rem 0;
        }
        .v22-summary-card {
            border: 1px solid color-mix(in srgb, var(--text-color, #111827) 10%, transparent);
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
            background: color-mix(in srgb, var(--secondary-background-color, #f8fafc) 76%, transparent);
            min-height: 100%;
        }
        .v22-summary-card-title {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: color-mix(in srgb, var(--text-color, #111827) 62%, white 38%);
            margin-bottom: 0.45rem;
        }
        .v22-summary-row {
            display: flex;
            justify-content: space-between;
            gap: 0.7rem;
            padding: 0.16rem 0;
            font-size: 0.9rem;
            line-height: 1.35;
        }
        .v22-summary-label {
            color: color-mix(in srgb, var(--text-color, #111827) 72%, white 28%);
        }
        .v22-summary-value {
            font-weight: 600;
            text-align: right;
            color: var(--text-color, #111827);
            overflow-wrap: anywhere;
        }
        .v22-notice {
            border-radius: 8px;
            padding: 0.75rem 0.9rem;
            margin: 0.35rem 0 0.9rem 0;
            font-size: 0.9rem;
            line-height: 1.4;
            border: 1px solid transparent;
        }
        .v22-notice-warning {
            background: color-mix(in srgb, #f59e0b 12%, var(--secondary-background-color, #f8fafc));
            border-color: color-mix(in srgb, #f59e0b 28%, transparent);
            color: var(--text-color, #111827);
        }
        .v22-notice-info {
            background: color-mix(in srgb, var(--primary-color, #2563eb) 10%, var(--secondary-background-color, #f8fafc));
            border-color: color-mix(in srgb, var(--primary-color, #2563eb) 22%, transparent);
            color: var(--text-color, #111827);
        }
        .v22-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.3rem 0 0.75rem 0;
        }
        .v22-chip {
            border-radius: 999px;
            padding: 0.18rem 0.65rem;
            font-size: 0.8rem;
            line-height: 1.3;
            border: 1px solid color-mix(in srgb, var(--text-color, #111827) 12%, transparent);
            background: color-mix(in srgb, var(--secondary-background-color, #f8fafc) 82%, transparent);
            color: var(--text-color, #111827);
        }
        .v22-progress-strip {
            border: 1px solid color-mix(in srgb, var(--text-color, #111827) 10%, transparent);
            border-radius: 8px;
            padding: 0.9rem 1rem;
            background: color-mix(in srgb, var(--secondary-background-color, #f8fafc) 70%, transparent);
            margin: 0.35rem 0 0.95rem 0;
        }
        .v22-progress-title {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: color-mix(in srgb, var(--text-color, #111827) 62%, white 38%);
            margin-bottom: 0.25rem;
        }
        .v22-progress-summary {
            font-size: 1rem;
            font-weight: 700;
            color: var(--text-color, #111827);
            margin-bottom: 0.65rem;
        }
        .v22-status-badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
        }
        .v22-status-badge {
            border-radius: 999px;
            padding: 0.22rem 0.7rem;
            font-size: 0.8rem;
            line-height: 1.3;
            border: 1px solid color-mix(in srgb, var(--text-color, #111827) 12%, transparent);
            background: color-mix(in srgb, var(--secondary-background-color, #f8fafc) 82%, transparent);
            color: var(--text-color, #111827);
        }
        .v22-status-badge-ready {
            background: color-mix(in srgb, #16a34a 11%, var(--secondary-background-color, #f8fafc));
            border-color: color-mix(in srgb, #16a34a 26%, transparent);
        }
        .v22-status-badge-review {
            background: color-mix(in srgb, #f59e0b 11%, var(--secondary-background-color, #f8fafc));
            border-color: color-mix(in srgb, #f59e0b 26%, transparent);
        }
        .v22-status-badge-stale {
            background: color-mix(in srgb, #2563eb 11%, var(--secondary-background-color, #f8fafc));
            border-color: color-mix(in srgb, #2563eb 26%, transparent);
        }
        .v22-status-badge-pending {
            background: color-mix(in srgb, var(--secondary-background-color, #f8fafc) 88%, transparent);
        }
        .v22-domain-header {
            border: 1px solid color-mix(in srgb, var(--text-color, #111827) 10%, transparent);
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            background: color-mix(in srgb, var(--secondary-background-color, #f8fafc) 62%, transparent);
            margin: 0.45rem 0 0.7rem 0;
        }
        .v22-domain-title-row {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 0.8rem;
        }
        .v22-domain-title {
            font-size: 1rem;
            font-weight: 700;
            color: var(--text-color, #111827);
            line-height: 1.25;
        }
        .v22-domain-subtitle {
            margin-top: 0.2rem;
            font-size: 0.9rem;
            color: color-mix(in srgb, var(--text-color, #111827) 72%, white 28%);
        }
        .v22-domain-lines {
            margin-top: 0.45rem;
            display: grid;
            gap: 0.18rem;
            font-size: 0.84rem;
            color: color-mix(in srgb, var(--text-color, #111827) 74%, white 26%);
        }
        .v22-reference-divider {
            display: grid;
            grid-template-columns: 1.35fr 0.65fr 2.8fr minmax(180px, 1fr);
            gap: 0.7rem;
            align-items: center;
            margin: 0.25rem 0 0.55rem 0;
        }
        .v22-reference-pill {
            border-radius: 999px;
            padding: 0.18rem 0.7rem;
            font-size: 0.78rem;
            line-height: 1.3;
            border: 1px solid color-mix(in srgb, var(--text-color, #111827) 10%, transparent);
            background: color-mix(in srgb, var(--secondary-background-color, #f8fafc) 80%, transparent);
            color: color-mix(in srgb, var(--text-color, #111827) 78%, white 22%);
        }
        .v22-reference-spacer {
            min-height: 0.1rem;
        }
        .v22-apply-result {
            border-radius: 8px;
            padding: 0.7rem 0.85rem;
            margin: 0.35rem 0 0.7rem 0;
            border: 1px solid transparent;
            font-size: 0.88rem;
            line-height: 1.35;
        }
        .v22-apply-result-ready {
            background: color-mix(in srgb, #16a34a 11%, var(--secondary-background-color, #f8fafc));
            border-color: color-mix(in srgb, #16a34a 26%, transparent);
        }
        .v22-apply-result-review {
            background: color-mix(in srgb, #f59e0b 11%, var(--secondary-background-color, #f8fafc));
            border-color: color-mix(in srgb, #f59e0b 26%, transparent);
        }
        .v22-apply-result-stale {
            background: color-mix(in srgb, #2563eb 11%, var(--secondary-background-color, #f8fafc));
            border-color: color-mix(in srgb, #2563eb 26%, transparent);
        }
        .v22-apply-result-pending {
            background: color-mix(in srgb, var(--secondary-background-color, #f8fafc) 82%, transparent);
            border-color: color-mix(in srgb, var(--text-color, #111827) 10%, transparent);
        }
        .v22-preview-strip {
            border: 1px solid color-mix(in srgb, var(--text-color, #111827) 10%, transparent);
            border-radius: 8px;
            padding: 0.9rem 1rem;
            background: color-mix(in srgb, var(--secondary-background-color, #f8fafc) 68%, transparent);
            margin: 0.35rem 0 0.9rem 0;
        }
        .v22-preview-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 0.7rem;
        }
        .v22-preview-item {
            border: 1px solid color-mix(in srgb, var(--text-color, #111827) 10%, transparent);
            border-radius: 8px;
            padding: 0.7rem 0.8rem;
            background: color-mix(in srgb, var(--secondary-background-color, #f8fafc) 82%, transparent);
        }
        .v22-preview-label {
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: color-mix(in srgb, var(--text-color, #111827) 62%, white 38%);
            margin-bottom: 0.2rem;
        }
        .v22-preview-value {
            font-size: 0.94rem;
            font-weight: 700;
            color: var(--text-color, #111827);
        }
        .v22-scenario-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 0.75rem;
            margin: 0.35rem 0 0.95rem 0;
        }
        .v22-scenario-card {
            border: 1px solid color-mix(in srgb, var(--text-color, #111827) 10%, transparent);
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            background: color-mix(in srgb, var(--secondary-background-color, #f8fafc) 76%, transparent);
        }
        .v22-scenario-title {
            font-size: 0.96rem;
            font-weight: 700;
            color: var(--text-color, #111827);
        }
        .v22-scenario-meta {
            margin-top: 0.18rem;
            font-size: 0.84rem;
            color: color-mix(in srgb, var(--text-color, #111827) 72%, white 28%);
        }
        .v22-scenario-section {
            margin-top: 0.55rem;
            font-size: 0.84rem;
            line-height: 1.35;
        }
        .v22-scenario-section strong {
            color: var(--text-color, #111827);
        }
        .v22-group-header {
            font-size: 0.92rem;
            font-weight: 700;
            color: var(--text-color, #111827);
            margin: 0.3rem 0 0.45rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_v22_step_header(step_payload: dict | None) -> None:
    payload = dict(step_payload or {})
    if not payload:
        return
    status_text = "Active" if payload.get("is_active") else _status_label(payload.get("base_status"))
    st.markdown(
        (
            '<div class="v22-step-header">'
            '<div class="v22-step-header-main">'
            f'<div class="v22-step-index">{int(payload.get("index") or 0)}</div>'
            "<div>"
            f'<div class="v22-step-title">{html.escape(str(payload.get("label") or ""))}</div>'
            f'<div class="v22-step-caption">{html.escape(str(payload.get("caption") or ""))}</div>'
            "</div>"
            "</div>"
            f'<div class="v22-step-status">{html.escape(str(payload.get("icon") or ""))} {html.escape(status_text)}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_v22_context_strip(items: list[dict] | None) -> None:
    cells = []
    for item in list(items or []):
        cells.append(
            '<div class="v22-context-item">'
            f'<div class="v22-context-label">{html.escape(str(item.get("label") or ""))}</div>'
            f'<div class="v22-context-value">{html.escape(str(item.get("value") or ""))}</div>'
            "</div>"
        )
    if not cells:
        return
    st.markdown('<div class="v22-context-strip">' + "".join(cells) + "</div>", unsafe_allow_html=True)


def render_v22_sidebar_step_meta(step_payload: dict | None) -> None:
    payload = dict(step_payload or {})
    if not payload:
        return
    st.markdown(
        (
            '<div class="v22-sidebar-meta">'
            f'<div class="v22-sidebar-summary">{html.escape(str(payload.get("summary") or ""))}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_v22_branding_header(payload: dict | None) -> None:
    data = dict(payload or {})
    if not data.get("loaded"):
        st.markdown(
            '<div class="v22-branding-empty">Load a baseline to populate brand and legislation context.</div>',
            unsafe_allow_html=True,
        )
        return

    make = str(data.get("make") or "")
    legislation = str(data.get("legislation") or "")
    context = {"make": make, "legislation": legislation}
    logo_path = data.get("logo_path") or search_logo(context, base_dir=str(_LOGOS_DIR), fallback="_unknown.png")
    legislation_icon_path = data.get("legislation_icon_path") or get_legislation_icon(context, base_dir=str(_IMAGES_DIR))

    col_logo, col_leg, col_text = st.columns([0.55, 0.55, 4.9])
    _render_branding_cell(col_logo, logo_path, make, width=56, empty_label="Make")
    _render_branding_cell(col_leg, legislation_icon_path, legislation, width=36, empty_label="Legislation")
    with col_text:
        st.markdown(
            (
                '<div class="v22-branding-label">VDE Request Builder</div>'
                f'<div class="v22-branding-value">VDE #{html.escape(str(data.get("baseline_id") or "-"))} &#8226; {html.escape(legislation or "Unknown legislation")} &#8226; {int(data.get("proposal_count") or 0)} proposals</div>'
                f'<div class="v22-branding-meta">{html.escape(make or "Unknown make")}</div>'
            ),
            unsafe_allow_html=True,
        )


def render_v22_summary_groups(groups: list[dict] | None) -> None:
    cards = []
    for group in list(groups or []):
        rows = []
        for item in list(dict(group).get("items") or []):
            rows.append(
                '<div class="v22-summary-row">'
                f'<div class="v22-summary-label">{html.escape(str(item.get("label") or ""))}</div>'
                f'<div class="v22-summary-value">{html.escape(str(item.get("value") or ""))}</div>'
                "</div>"
            )
        cards.append(
            '<div class="v22-summary-card">'
            f'<div class="v22-summary-card-title">{html.escape(str(group.get("title") or ""))}</div>'
            + "".join(rows)
            + "</div>"
        )
    if not cards:
        return
    st.markdown('<div class="v22-summary-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def render_v22_notice_strip(message: str, *, tone: str = "info") -> None:
    text = str(message or "").strip()
    if not text:
        return
    tone_class = "v22-notice-warning" if tone == "warning" else "v22-notice-info"
    st.markdown(f'<div class="v22-notice {tone_class}">{html.escape(text)}</div>', unsafe_allow_html=True)


def render_v22_chip_list(chips: list[str] | None) -> None:
    values = [str(item).strip() for item in list(chips or []) if str(item).strip()]
    if not values:
        return
    st.markdown(
        '<div class="v22-chip-row">' + "".join(f'<div class="v22-chip">{html.escape(item)}</div>' for item in values) + "</div>",
        unsafe_allow_html=True,
    )


def render_v22_request_inputs_overview(payload: dict | None) -> None:
    data = dict(payload or {})
    st.markdown(
        (
            '<div class="v22-progress-strip">'
            f'<div class="v22-progress-summary">{html.escape(str(data.get("summary") or "0 direct domains | 0 applied"))}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_v22_domain_card_header(payload: dict | None) -> None:
    data = dict(payload or {})
    status_badge = _status_badge_html(str(data.get("status_label") or "Pending"), "", str(data.get("status_tone") or "pending"))
    st.markdown(
        (
            '<div class="v22-domain-header">'
            '<div class="v22-domain-title-row">'
            '<div>'
            f'<div class="v22-domain-title">{html.escape(str(data.get("label") or ""))}</div>'
            f'<div class="v22-domain-subtitle">{html.escape(str(data.get("proposal_type_summary") or ""))}</div>'
            "</div>"
            + status_badge
            + "</div>"
            f'<div class="v22-domain-subtitle">{html.escape(str(data.get("active_proposal_count") or 0))} proposal(s)</div>'
            + "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_v22_reference_divider() -> None:
    st.markdown(
        (
            '<div class="v22-reference-divider">'
            '<div class="v22-reference-spacer"></div>'
            '<div class="v22-reference-spacer"></div>'
            '<div class="v22-reference-pill">Reference baseline</div>'
            '<div class="v22-reference-pill">Requested proposals</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_v22_apply_result(message: str, *, tone: str = "pending") -> None:
    text = str(message or "").strip()
    if not text:
        return
    tone_class = {
        "ready": "v22-apply-result-ready",
        "review": "v22-apply-result-review",
        "stale": "v22-apply-result-stale",
        "pending": "v22-apply-result-pending",
    }.get(str(tone or "pending"), "v22-apply-result-pending")
    st.markdown(f'<div class="v22-apply-result {tone_class}">{html.escape(text)}</div>', unsafe_allow_html=True)


def render_v22_preview_status_strip(items: list[dict] | None) -> None:
    cells = []
    for item in list(items or []):
        cells.append(
            '<div class="v22-preview-item">'
            f'<div class="v22-preview-label">{html.escape(str(item.get("label") or ""))}</div>'
            f'<div class="v22-preview-value">{html.escape(str(item.get("value") or ""))}</div>'
            "</div>"
        )
    if not cells:
        return
    st.markdown('<div class="v22-preview-strip"><div class="v22-preview-grid">' + "".join(cells) + "</div></div>", unsafe_allow_html=True)


def render_v22_scenario_overview_cards(cards: list[dict] | None) -> None:
    items = []
    for card in list(cards or []):
        metrics = []
        for item in list(card.get("metrics") or []):
            metrics.append(
                '<div class="v22-scenario-section">'
                f'<strong>{html.escape(str(item.get("label") or ""))}:</strong> '
                f'{html.escape(str(item.get("value") or "-"))}'
                "</div>"
            )
        sections = []
        for label, values in (
            ("Changes", card.get("changes")),
            ("Inherited", card.get("inherited")),
            ("Not used", card.get("not_used")),
            ("Review", card.get("review")),
            ("Missing", card.get("missing")),
        ):
            current = [str(item).strip() for item in list(values or []) if str(item).strip()]
            if not current:
                continue
            sections.append(f'<div class="v22-scenario-section"><strong>{html.escape(label)}:</strong> {html.escape(", ".join(current))}</div>')
        cycle_rows = []
        for cycle in list(card.get("cycle_results") or []):
            values = []
            if cycle.get("total") not in (None, ""):
                values.append(f'TOTAL {_format_compact_number(cycle.get("total"))}')
            if cycle.get("net") not in (None, ""):
                values.append(f'NET {_format_compact_number(cycle.get("net"))}')
            if values:
                cycle_rows.append(
                    '<div class="v22-scenario-section">'
                    f'<strong>{html.escape(str(cycle.get("label") or ""))}:</strong> {html.escape(" | ".join(values))}'
                    "</div>"
                )
        items.append(
            '<div class="v22-scenario-card">'
            f'<div class="v22-scenario-title">{html.escape(str(card.get("label") or ""))}</div>'
            + (
                f'<div class="v22-scenario-meta">{html.escape(str(card.get("reference_id") or ""))}</div>'
                if card.get("label") == "Baseline" and card.get("reference_id")
                else f'<div class="v22-scenario-meta">From: {html.escape(str(card.get("walk_from") or ""))}</div>'
            )
            + f'<div class="v22-scenario-meta">Status: {html.escape(str(card.get("status") or ""))}</div>'
            + "".join(metrics)
            + ("<div class=\"v22-scenario-section\"><strong>VDE by cycle</strong></div>" + "".join(cycle_rows) if cycle_rows else "")
            + "".join(sections)
            + "</div>"
        )
    if not items:
        return
    st.markdown('<div class="v22-scenario-grid">' + "".join(items) + "</div>", unsafe_allow_html=True)


def _format_compact_number(value) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def render_v22_group_header(title: str) -> None:
    text = str(title or "").strip()
    if not text:
        return
    st.markdown(f'<div class="v22-group-header">{html.escape(text)}</div>', unsafe_allow_html=True)


def _render_branding_cell(column, image_path: str | None, label: str, *, width: int, empty_label: str) -> None:
    with column:
        rendered = render_inline_image(image_path, width=width, caption=label or empty_label)
        if not rendered:
            st.markdown(
                f'<div class="v22-branding-empty">{html.escape(label or empty_label)}</div>',
                unsafe_allow_html=True,
            )


def _status_label(status: str | None) -> str:
    value = str(status or "pending").replace("_", " ").strip()
    return value[:1].upper() + value[1:] if value else "Pending"


def _status_badge_html(label: str, value, tone: str) -> str:
    suffix = f" {value}" if value not in ("", None) else ""
    return f'<div class="v22-status-badge v22-status-badge-{html.escape(str(tone or "pending"))}">{html.escape(str(label))}{html.escape(str(suffix))}</div>'
