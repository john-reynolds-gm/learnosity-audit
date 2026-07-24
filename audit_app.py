"""
UC3 Alignment Audit Review

Standalone Streamlit app for reviewing Claude's Step D confidence-scoring
verdicts on existing EM2-to-CCSS crosswalk tags. Separate from the main
EM2 Standards Alignment Explorer app on purpose -- that app is a browse
tool for exploring already-trusted data; this one is a review/triage tool
for deciding which tags need attention, and its data (verdicts, evidence,
rationale) doesn't belong alongside a general-purpose browser.

Data inputs:
    scored_alignments.csv       - output of score_alignments.py's `collect` command
                                   (state, anchor_id, existing_ccss_id, verdict,
                                   confidence, recommended_ccss_id, rationale)
    step_d_input.csv           - the full evidence file fed into Step D
                                   (existing tag + top-5 candidates, with
                                   lesson_overlap_pct / bi_score for each)
    all_states_standards_k8.csv - state standard reference (id + text), used to
                                   find standards with zero existing tags
    all_states_k8.csv           - long-format state standard/lesson pairs, used
                                   to count EM2 lessons already aligned to a
                                   standard that has no Learnosity tag
    existing_crosswalk_export.csv - re-read here (independent of step_d_input.csv)
                                   to get every state standard that has AT LEAST
                                   ONE existing tag, since step_d_input.csv only
                                   covers tags that made it into Step D
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from urllib.parse import quote_plus

from standard_id_utils import isolate_standard_id, normalize_separators

# ── CONFIG ────────────────────────────────────────────────────────────────────

SCORED_CSV = "scored_alignments.csv"
EVIDENCE_CSV = "step_d_input.csv"
STANDARDS_CSV = "all_states_standards_k8.csv"
LESSONS_CSV = "all_states_k8.csv"
CROSSWALK_CSV = "existing_crosswalk_export.csv"

VERDICT_ORDER = ["likely_incorrect", "uncertain", "likely_correct", "confirmed"]
VERDICT_COLORS = {
    "likely_incorrect": "🔴",
    "uncertain": "🟡",
    "likely_correct": "🟢",
    "confirmed": "✅",
}

ALL_US_STATES = [
    "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD",
    "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH",
    "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY",
]

st.set_page_config(
    page_title="UC3 Alignment Audit Review",
    page_icon="🔍",
    layout="wide",
)


# ── DATA LOADING ──────────────────────────────────────────────────────────────

@st.cache_data
def load_scored() -> pd.DataFrame:
    return pd.read_csv(SCORED_CSV, dtype={"state": str, "anchor_id": str, "existing_ccss_id": str})


@st.cache_data
def load_evidence() -> pd.DataFrame:
    return pd.read_csv(EVIDENCE_CSV)


@st.cache_data
def build_triage_df() -> pd.DataFrame:
    """
    One row per judged tag: verdict info joined with the anchor's own text/
    grade and the existing tag's own lesson-overlap evidence (both live in
    step_d_input.csv, not in scored_alignments.csv itself).
    """
    scored = load_scored()
    evidence = load_evidence()

    existing = evidence[evidence["source"] == "existing"]
    existing_evidence = existing.rename(columns={"ccss_id": "existing_ccss_id"})[
        ["state", "anchor_id", "existing_ccss_id", "anchor_text", "anchor_grade",
         "ccss_text", "lesson_overlap_pct", "in_top5_candidates", "candidate_rank_if_any"]
    ].rename(columns={"ccss_text": "existing_ccss_text",
                       "lesson_overlap_pct": "existing_lesson_overlap_pct"})

    merged = scored.merge(
        existing_evidence,
        on=["state", "anchor_id", "existing_ccss_id"],
        how="left",
    )
    return merged


@st.cache_data
def load_standards() -> pd.DataFrame:
    """Standard-level rows only (drop domain/cluster heading rows)."""
    df = pd.read_csv(STANDARDS_CSV, dtype=str)
    return df[df["level"] == "standard"].dropna(subset=["standard_id"]).reset_index(drop=True)


@st.cache_data
def load_lesson_counts() -> pd.DataFrame:
    """
    Per (state, standard_id) count of EM2 lessons already aligned, from the
    long-format lesson file. Standards with no lesson_id at all never appear
    here -- treat a missing lookup key as a count of 0.
    """
    df = pd.read_csv(LESSONS_CSV, dtype=str)
    has_lesson = df["lesson_id"].notna() & (df["lesson_id"] != "")
    counts = (
        df[has_lesson]
        .groupby(["state", "standard_id"])
        .size()
        .reset_index(name="lesson_count")
    )
    return counts


@st.cache_data
def load_tagged_ids() -> set:
    """
    Set of (state, normalized_standard_id) pairs that have at least one
    existing Learnosity tag, from the crosswalk export. The crosswalk's
    state_standard_id is already normalized (isolate_standard_id +
    normalize_separators applied in build_database.py), so the standards
    file side needs the same normalization to compare on equal footing --
    same approach analyze_id_mismatches.py uses.
    """
    df = pd.read_csv(CROSSWALK_CSV, dtype=str)
    return set(zip(df["state"], df["state_standard_id"]))


@st.cache_data
def build_state_summary_df() -> pd.DataFrame:
    """
    One row per state: total judged tags, count/% per verdict, mean
    confidence. This is the entry point for triage -- which states need
    attention before drilling into individual standards.
    """
    scored = load_scored()
    total = scored.groupby("state").size().rename("total")
    mean_conf = scored.groupby("state")["confidence"].mean().rename("mean_confidence")

    verdict_counts = (
        scored.groupby(["state", "verdict"]).size().unstack(fill_value=0)
    )
    for v in VERDICT_ORDER:
        if v not in verdict_counts.columns:
            verdict_counts[v] = 0
    verdict_counts = verdict_counts[VERDICT_ORDER]

    summary = pd.concat([total, verdict_counts, mean_conf], axis=1).reset_index()
    summary["pct_flagged"] = (
        (summary["uncertain"] + summary["likely_incorrect"]) / summary["total"]
    )
    return summary.sort_values("pct_flagged", ascending=False).reset_index(drop=True)


@st.cache_data
def build_no_tag_gaps_df() -> pd.DataFrame:
    """
    Standards with real EM2 lesson content but zero existing Learnosity
    tags at all -- never scored by Step D since there was no tag to judge,
    so this is a separate gap list, not a verdict.
    """
    standards = load_standards().copy()
    standards["standard_id_norm"] = (
        standards["standard_id"].apply(isolate_standard_id).apply(normalize_separators)
    )
    tagged = load_tagged_ids()
    standards["has_tag"] = list(
        zip(standards["state"], standards["standard_id_norm"])
    )
    standards["has_tag"] = standards["has_tag"].isin(tagged)

    gaps = standards[~standards["has_tag"]].copy()

    lesson_counts = load_lesson_counts()
    gaps = gaps.merge(lesson_counts, on=["state", "standard_id"], how="left")
    gaps["lesson_count"] = gaps["lesson_count"].fillna(0).astype(int)

    return gaps[["state", "standard_id", "grade", "standard_text", "lesson_count"]].sort_values(
        ["state", "lesson_count"], ascending=[True, False]
    ).reset_index(drop=True)


def get_candidates(state: str, anchor_id: str) -> pd.DataFrame:
    evidence = load_evidence()
    return evidence[
        (evidence["state"] == state) & (evidence["anchor_id"] == anchor_id)
        & (evidence["source"] == "candidate")
    ].sort_values("rank")


def get_other_existing(state: str, anchor_id: str, exclude_ccss_id: str) -> pd.DataFrame:
    evidence = load_evidence()
    return evidence[
        (evidence["state"] == state) & (evidence["anchor_id"] == anchor_id)
        & (evidence["source"] == "existing") & (evidence["ccss_id"] != exclude_ccss_id)
    ]


# ── DISPLAY HELPERS ───────────────────────────────────────────────────────────

def verdict_label(v: str) -> str:
    return f"{VERDICT_COLORS.get(v, '')} {v}"


def build_jump_link(state: str, anchor_id: str, existing_ccss_id: str, label: str) -> str:
    """
    A real HTML link, not a double-click handler -- double-click on a plain
    HTML table can't report back to Streamlit's Python side without a custom
    JS component, which is a lot of extra machinery for one interaction.
    A link achieves the same "click a standard, jump to its detail view"
    outcome: it encodes state + anchor + which specific existing tag (a
    standard can have more than one), sets those as URL query params, and
    the page scrolls to the #drilldown anchor on load.
    """
    qs = (
        f"jump_state={quote_plus(str(state))}"
        f"&jump_anchor={quote_plus(str(anchor_id))}"
        f"&jump_ccss={quote_plus(str(existing_ccss_id))}"
    )
    return f'<a href="?{qs}#drilldown" target="_self">{label}</a>'


def render_wrapped_table(df: pd.DataFrame, height: str = "480px", raw_html_columns: list | None = None) -> None:
    """
    st.dataframe has no built-in option to wrap long cell text -- it always
    truncates regardless of column width (open Streamlit limitation,
    streamlit/streamlit#5386). Renders a scrollable HTML table instead, with
    CSS word-wrap so standard/rationale text is fully visible in place.
    Loses st.dataframe's column_config niceties (progress bars, sorting UI)
    in exchange for readable text -- worth it for these two text-heavy tables.

    raw_html_columns: columns whose values are already-built HTML (e.g. the
    jump-link anchor tags) and should NOT be escaped, unlike every other
    column, which is escaped by default to render safely as plain text.
    """
    styled = df.style.format(na_rep="—", escape="html")
    if raw_html_columns:
        styled = styled.format(escape=None, subset=raw_html_columns)
    styled = (
        styled
        .set_table_styles([
            {"selector": "th", "props": [
                ("position", "sticky"), ("top", "0"), ("background", "#f0f2f6"),
                ("text-align", "left"), ("padding", "6px 10px"), ("font-size", "0.85rem"),
                ("z-index", "1"),
            ]},
            {"selector": "td", "props": [
                ("text-align", "left"), ("padding", "6px 10px"), ("vertical-align", "top"),
                ("font-size", "0.85rem"), ("white-space", "normal"), ("word-break", "break-word"),
            ]},
            {"selector": "table", "props": [("width", "100%"), ("border-collapse", "collapse")]},
        ])
        .hide(axis="index")
    )
    st.markdown(
        f'<div style="max-height:{height}; overflow-y:auto; '
        f'border:1px solid #e6e6e6; border-radius:6px;">{styled.to_html()}</div>',
        unsafe_allow_html=True,
    )


def build_risk_map(triage_df: pd.DataFrame) -> go.Figure:
    """US choropleth: share of judged tags per state that are uncertain or
    likely_incorrect, as a quick visual entry point before the table."""
    risky = triage_df["verdict"].isin(["uncertain", "likely_incorrect"])
    per_state = triage_df.groupby("state").agg(total=("verdict", "count"), risky=("verdict", lambda s: risky.loc[s.index].sum()))
    per_state["pct"] = per_state["risky"] / per_state["total"]
    pct_map = per_state["pct"].to_dict()

    z_values = [pct_map.get(s, 0) for s in ALL_US_STATES]
    hover = [
        f"<b>{s}</b><br>{pct_map[s]:.0%} flagged" if s in pct_map else f"<b>{s}</b><br>No data"
        for s in ALL_US_STATES
    ]
    fig = go.Figure(go.Choropleth(
        locations=ALL_US_STATES, z=z_values, locationmode="USA-states",
        colorscale=[[0, "#E8E8E8"], [1, "#C0392B"]], showscale=False,
        marker_line_color="white", marker_line_width=0.5,
        hovertext=hover, hoverinfo="text",
    ))
    fig.update_layout(
        geo_scope="usa", margin=dict(l=0, r=0, t=0, b=0), height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        geo=dict(bgcolor="rgba(0,0,0,0)", lakecolor="rgba(0,0,0,0)"),
    )
    return fig


# ── APP ───────────────────────────────────────────────────────────────────────

st.title("🔍 UC3 Alignment Audit Review")
st.caption(
    "Reviewing Claude's confidence verdicts on existing EM2 → CCSS crosswalk tags. "
    "A low-confidence or 'likely_incorrect' verdict means the tag lacks strong supporting "
    "evidence -- not a guarantee it's wrong. Check the evidence in the detail view before "
    "acting on any single row."
)

triage_df = build_triage_df()

# ── Per-state summary ──
st.subheader("Per-state summary")
st.caption("Sorted by % flagged (uncertain + likely_incorrect) -- start here to decide which states to drill into.")

col_summary, col_map = st.columns([3, 2])

state_summary = build_state_summary_df().copy()
state_summary["pct_flagged"] = state_summary["pct_flagged"] * 100

with col_summary:
    summary_display = state_summary.rename(columns={
        "state": "State",
        "total": "Total",
        "likely_incorrect": "Likely Incorrect",
        "uncertain": "Uncertain",
        "likely_correct": "Likely Correct",
        "confirmed": "Confirmed",
        "mean_confidence": "Mean Confidence",
        "pct_flagged": "% Flagged",
    })
    st.dataframe(
        summary_display,
        width="stretch",
        hide_index=True,
        column_config={
            "% Flagged": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
            "Mean Confidence": st.column_config.NumberColumn(format="%.0f"),
        },
    )
    st.download_button(
        "Download per-state summary as CSV",
        data=summary_display.to_csv(index=False).encode("utf-8"),
        file_name="uc3_state_summary.csv",
        mime="text/csv",
    )

with col_map:
    st.plotly_chart(build_risk_map(triage_df), width="stretch")

st.divider()

# ── Filters ──
col_f1, col_f2, col_f3 = st.columns([2, 2, 3])

with col_f1:
    states = sorted(triage_df["state"].dropna().unique())
    selected_states = st.multiselect("State", states, default=[])

with col_f2:
    selected_verdicts = st.multiselect(
        "Verdict", VERDICT_ORDER,
        default=["likely_incorrect", "uncertain"],
    )

with col_f3:
    conf_range = st.slider("Confidence range", 0, 100, (0, 100))

filtered = triage_df.copy()
if selected_states:
    filtered = filtered[filtered["state"].isin(selected_states)]
if selected_verdicts:
    filtered = filtered[filtered["verdict"].isin(selected_verdicts)]
filtered = filtered[
    filtered["confidence"].between(conf_range[0], conf_range[1]) | filtered["confidence"].isna()
]

st.caption(f"{len(filtered):,} of {len(triage_df):,} judged tags shown")

# ── Triage table ──
sort_order = {v: i for i, v in enumerate(VERDICT_ORDER)}
filtered = filtered.assign(_sort=filtered["verdict"].map(sort_order)).sort_values(
    ["_sort", "confidence"], ascending=[True, True]
).drop(columns="_sort")

display = filtered[[
    "state", "anchor_id", "anchor_text", "existing_ccss_id", "verdict",
    "confidence", "recommended_ccss_id", "rationale",
]].copy()
display["verdict"] = display["verdict"].apply(verdict_label)

csv_export = display.rename(columns={
    "anchor_id": "Standard",
    "anchor_text": "Standard Text",
    "existing_ccss_id": "Existing Tag",
    "verdict": "Verdict",
    "confidence": "Confidence",
    "recommended_ccss_id": "Recommended",
    "rationale": "Rationale",
    "state": "State",
})

# On-screen only: replace the plain standard id with a clickable link that
# jumps to and pre-selects that exact row in the drill-down section below.
render_display = display.copy()
render_display["anchor_id"] = [
    build_jump_link(r["state"], r["anchor_id"], r["existing_ccss_id"], r["anchor_id"])
    for _, r in display.iterrows()
]
render_display = render_display.rename(columns={
    "anchor_id": "Standard",
    "anchor_text": "Standard Text",
    "existing_ccss_id": "Existing Tag",
    "verdict": "Verdict",
    "confidence": "Confidence",
    "recommended_ccss_id": "Recommended",
    "rationale": "Rationale",
    "state": "State",
})

render_wrapped_table(render_display, raw_html_columns=["Standard"])

st.download_button(
    "Download filtered results as CSV",
    data=csv_export.to_csv(index=False).encode("utf-8"),
    file_name="uc3_triage_filtered.csv",
    mime="text/csv",
)

# ── Drill-down ──
st.divider()
st.markdown('<div id="drilldown"></div>', unsafe_allow_html=True)
st.subheader("Inspect a standard")


def render_drill_down(row: dict) -> None:
    state, anchor_id = row["state"], row["anchor_id"]

    with st.container(border=True):
        st.markdown(f"**{state} grade {row.get('anchor_grade', '—')} · {anchor_id}**")
        st.write(row.get("anchor_text", "*No text available.*"))

    col_existing, col_verdict = st.columns(2)

    with col_existing:
        st.markdown("**Tag being evaluated**")
        st.write(f"`{row['existing_ccss_id']}` — {row.get('existing_ccss_text', '')}")
        overlap = row.get("existing_lesson_overlap_pct")
        st.write(f"Lesson overlap: **{overlap:.0%}**" if pd.notna(overlap) else "Lesson overlap: n/a")
        in_top5 = row.get("in_top5_candidates")
        rank_note = f" (rank {int(row['candidate_rank_if_any'])})" if pd.notna(row.get("candidate_rank_if_any")) else ""
        st.write(f"In top-5 candidates: {'Yes' + rank_note if in_top5 else 'No'}")

        other = get_other_existing(state, anchor_id, row["existing_ccss_id"])
        if len(other):
            st.markdown("*Other existing tags for this standard:*")
            for _, o in other.iterrows():
                st.write(f"- `{o['ccss_id']}` (overlap {o['lesson_overlap_pct']:.0%})")

    with col_verdict:
        st.markdown("**Claude's verdict**")
        st.write(f"{verdict_label(row['verdict'])}  ·  confidence {int(row['confidence']) if pd.notna(row['confidence']) else '—'}")
        st.write(f"Recommended: `{row.get('recommended_ccss_id', '—')}`")
        st.write(row.get("rationale", ""))

    st.markdown("**Top-5 similarity candidates**")
    cands = get_candidates(state, anchor_id)
    if len(cands):
        cand_display = cands[["rank", "ccss_id", "ccss_text", "bi_score", "lesson_overlap_pct"]].rename(
            columns={"rank": "Rank", "ccss_id": "CCSS ID", "ccss_text": "Text",
                     "bi_score": "Similarity", "lesson_overlap_pct": "Lesson Overlap"}
        )
        st.dataframe(
            cand_display, width="stretch", hide_index=True,
            column_config={
                "Text": st.column_config.TextColumn(width="large"),
                "Lesson Overlap": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
            },
        )
    else:
        st.write("No candidates recorded for this standard.")


qp = st.query_params
jump_state, jump_anchor, jump_ccss = qp.get("jump_state"), qp.get("jump_anchor"), qp.get("jump_ccss")
jump_row = None
if jump_state and jump_anchor and jump_ccss:
    match = triage_df[
        (triage_df["state"] == jump_state)
        & (triage_df["anchor_id"] == jump_anchor)
        & (triage_df["existing_ccss_id"] == jump_ccss)
    ]
    if len(match):
        jump_row = match.iloc[0].to_dict()

if jump_row is not None:
    st.caption(
        f"Jumped here from the table above: **{jump_row['state']} {jump_row['anchor_id']}** "
        f"(tag `{jump_row['existing_ccss_id']}`)."
    )
    if st.button("Clear and pick from the filtered list instead"):
        st.query_params.clear()
        st.rerun()
    render_drill_down(jump_row)
elif filtered.empty:
    st.info("No rows match the current filters.")
else:
    labels = [
        f"{r['state']} {r['anchor_id']} — {verdict_label(r['verdict'])} ({int(r['confidence']) if pd.notna(r['confidence']) else '—'})"
        for _, r in filtered.iterrows()
    ]
    row_by_label = dict(zip(labels, filtered.to_dict("records")))
    chosen = st.selectbox("Pick a flagged standard to see the full evidence", labels)

    if chosen:
        render_drill_down(row_by_label[chosen])


# ── No tag but has content ──────────────────────────────────────────────────

st.divider()
st.subheader("Standards with no existing tag")
st.caption(
    "These standards were never scored by Step D -- there was no existing tag to judge. "
    "A standard with several EM2 lessons already aligned but still no Learnosity tag is a "
    "genuine coverage gap worth tagging, not a verdict to review."
)

gaps_df = build_no_tag_gaps_df()

col_g1, col_g2 = st.columns([2, 2])

with col_g1:
    gap_states = sorted(gaps_df["state"].dropna().unique())
    selected_gap_states = st.multiselect("State", gap_states, default=[], key="gap_states")

with col_g2:
    min_lessons = st.slider(
        "Minimum EM2 lessons aligned", 0, int(gaps_df["lesson_count"].max() or 0), 1,
        help="Filter to standards with at least this many lessons already aligned in EM2 -- "
             "higher values surface the more suspicious gaps first.",
    )

gaps_filtered = gaps_df[gaps_df["lesson_count"] >= min_lessons]
if selected_gap_states:
    gaps_filtered = gaps_filtered[gaps_filtered["state"].isin(selected_gap_states)]

n_no_content = (gaps_df["lesson_count"] == 0).sum()
st.caption(
    f"{len(gaps_filtered):,} of {len(gaps_df):,} untagged standard(s) shown  ·  "
    f"{n_no_content:,} of the {len(gaps_df):,} total have zero EM2 lessons aligned "
    f"(likely nothing to tag, not shown by default)"
)

gaps_display = gaps_filtered.rename(columns={
    "state": "State",
    "standard_id": "Standard",
    "grade": "Grade",
    "standard_text": "Text",
    "lesson_count": "EM2 Lessons Aligned",
}).fillna("—")

render_wrapped_table(gaps_display)

st.download_button(
    "Download gap list as CSV",
    data=gaps_display.to_csv(index=False).encode("utf-8"),
    file_name="uc3_no_tag_gaps.csv",
    mime="text/csv",
)