"""Streamlit dashboard for practicum evaluation intelligence.

Run from the project root:
    streamlit run app.py
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config import (
    agency_profiles_file,
    agency_trends_file,
    app_title,
    concern_fit_score,
    concern_recommendation,
    evaluations_text_file,
)
from constants import theme_labels
from theme_lexicon import theme_dictionary

# -----------------------------------------------------------------------------
# palette
# -----------------------------------------------------------------------------
C_CONCERN = "#B34F5B"
C_CONCERN_DEEP = "#8F3A46"
C_CAUTION = "#A65C28"
C_NEUTRAL = "#869299"
C_POSITIVE = "#5F89A3"
C_LIGHT_BLUE = "#77A8CA"
C_MUTED_BLUE = "#B4CFE0"
C_TEAL = "#4A6F64"
C_BG = "#ffffff"
C_SURFACE = "#f8f9fa"
C_BORDER = "#e5e7eb"
C_SPINE = "#c7cfd4"
C_TITLE = "#1c1e21"
C_TEXT = "#4b555b"
C_SHADE = "rgba(241,224,225,0.65)"


# -----------------------------------------------------------------------------
# page config and css
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=app_title,
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_CSS = f"""
<style>
    .block-container {{
        padding-top: 1.1rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }}
    div[data-testid="metric-container"] {{
        background-color: {C_SURFACE};
        border: 1px solid {C_BORDER};
        border-radius: 10px;
        padding: 16px 20px;
    }}
    .badge-flag {{
        display: inline-block;
        background: #fdecea;
        color: {C_CONCERN};
        border: 1px solid #f5c6c2;
        border-radius: 5px;
        padding: 3px 10px;
        font-size: 0.8rem;
        font-weight: 600;
    }}
    .badge-ok {{
        display: inline-block;
        background: #e8f1f9;
        color: {C_POSITIVE};
        border: 1px solid #b8d4ea;
        border-radius: 5px;
        padding: 3px 10px;
        font-size: 0.8rem;
        font-weight: 600;
    }}
    .badge-mismatch {{
        display: inline-block;
        background: #fef3e2;
        color: {C_CAUTION};
        border: 1px solid #f9d5a0;
        border-radius: 5px;
        padding: 3px 10px;
        font-size: 0.8rem;
        font-weight: 600;
    }}
    .warn-box {{
        background: #fff8f0;
        border-left: 4px solid {C_CAUTION};
        padding: 0.75rem 1rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.88rem;
        color: #444;
        margin-bottom: 1rem;
    }}
    .detail-card {{
        background: {C_SURFACE};
        border: 1px solid {C_BORDER};
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }}
    .detail-label {{
        font-size: 0.82rem;
        color: {C_NEUTRAL};
        margin-bottom: 1px;
    }}
    .detail-value {{
        font-size: 1.05rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.7rem;
    }}
    .detail-value-concern {{
        font-size: 1.05rem;
        font-weight: 600;
        color: {C_CONCERN};
        margin-bottom: 0.7rem;
    }}
    .hero-block {{
        background: {C_SURFACE};
        border: 1px solid {C_BORDER};
        border-radius: 12px;
        padding: 1.6rem 1.8rem 1.4rem 1.8rem;
        margin: 1rem 0 0.6rem 0;
    }}
    .hero-eyebrow {{
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {C_NEUTRAL};
        font-weight: 600;
        margin-bottom: 0.35rem;
    }}
    .hero-title {{
        font-size: 1.35rem;
        font-weight: 700;
        color: {C_TITLE};
        margin-bottom: 1.2rem;
        letter-spacing: -0.01em;
    }}
    .hero-row {{
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 1.4rem;
    }}
    .hero-number {{
        font-size: 2.4rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 0.35rem;
        letter-spacing: -0.02em;
    }}
    .hero-number-concern {{
        color: {C_CONCERN};
    }}
    .hero-number-caution {{
        color: {C_CAUTION};
    }}
    .hero-headline {{
        font-size: 0.98rem;
        font-weight: 600;
        color: {C_TITLE};
        margin-bottom: 0.35rem;
        line-height: 1.3;
    }}
    .hero-detail {{
        font-size: 0.85rem;
        color: {C_TEXT};
        line-height: 1.45;
    }}
    .chapter-eyebrow {{
        font-size: 0.72rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: {C_NEUTRAL};
        font-weight: 700;
        margin-top: 2.8rem;
        margin-bottom: 0.25rem;
    }}
    .chapter-heading {{
        font-size: 1.35rem;
        font-weight: 700;
        color: {C_TITLE};
        margin-bottom: 0.5rem;
        letter-spacing: -0.01em;
        line-height: 1.25;
    }}
    .chapter-bridge {{
        font-size: 0.92rem;
        color: {C_TEXT};
        font-style: italic;
        margin-bottom: 1.2rem;
        line-height: 1.5;
        max-width: 820px;
    }}
    .quick-stats-label {{
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: {C_NEUTRAL};
        font-weight: 600;
        margin-top: 0.4rem;
        margin-bottom: 0.6rem;
    }}
    div[data-baseweb="slider"] div[role="slider"] {{
        background-color: {C_CONCERN} !important;
        border-color: {C_CONCERN} !important;
    }}
    div[data-baseweb="slider"] > div > div > div > div {{
        background: {C_CONCERN} !important;
    }}
    div[data-baseweb="radio"] input[type="radio"]:checked + div {{
        border-color: {C_CONCERN} !important;
        background-color: {C_CONCERN} !important;
    }}
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="base-input"]:focus-within {{
        border-color: {C_CONCERN} !important;
        box-shadow: 0 0 0 1px {C_CONCERN} !important;
    }}
    div[data-baseweb="select"]:focus-within > div {{
        border-color: {C_CONCERN} !important;
        box-shadow: 0 0 0 1px {C_CONCERN} !important;
    }}
</style>
"""

st.markdown(APP_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# loading
# -----------------------------------------------------------------------------
@st.cache_data
def load_profiles() -> pd.DataFrame:
    """Load and standardize the agency profiles table."""
    df = pd.read_csv(agency_profiles_file)

    numeric_cols = [
        "placement_quality_score",
        "concern_indicator_count",
        "overall_sentiment_score",
        "recommendation_rate",
        "response_count",
        "mean_competency_score",
        "mean_response_length",
        *theme_labels.keys(),
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["recommendation_rate_pct"] = df["recommendation_rate"] * 100
    df["concern_flag"] = df["concern_flag"].fillna("no flag")
    df["misalignment_flag"] = df["misalignment_flag"].fillna("no flag")
    df["fit_trend"] = df["fit_trend"].fillna("stable")
    df["is_flagged"] = df["concern_flag"].str.lower().eq("review recommended")
    df["is_misaligned"] = df["misalignment_flag"].str.lower().eq(
        "score-narrative mismatch"
    )

    return df.sort_values(
        ["is_flagged", "placement_quality_score", "agency_name_display"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


@st.cache_data
def load_text_data() -> pd.DataFrame:
    """Load the saved evaluation text export."""
    return pd.read_csv(evaluations_text_file)


@st.cache_data
def load_trend_data() -> pd.DataFrame:
    """Load the saved agency-year trend output."""
    df = pd.read_csv(agency_trends_file)

    numeric_cols = [
        "academic_year_start",
        "placement_quality_score",
        "concern_indicator_count",
        "overall_sentiment_score",
        "recommendation_rate",
        "response_count",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["recommendation_rate_pct"] = df["recommendation_rate"] * 100
    df["concern_flag"] = df["concern_flag"].fillna("no flag")
    df["is_flagged"] = df["concern_flag"].str.lower().eq("review recommended")

    return df.sort_values(["academic_year_start", "agency_name_display"]).reset_index(
        drop=True
    )


# -----------------------------------------------------------------------------
# shared plot helpers
# -----------------------------------------------------------------------------
def base_layout(
    fig: go.Figure,
    title: str,
    subtitle: str | None = None,
) -> go.Figure:
    """Apply one shared Plotly layout."""
    title_text = f"<b>{title}</b>"
    if subtitle:
        title_text += (
            f"<br><span style='font-size:11px;color:{C_NEUTRAL}'>{subtitle}</span>"
        )

    fig.update_layout(
        font=dict(family="sans-serif", size=12, color="#1a1a1a"),
        legend=dict(
            orientation="h",
            x=1,
            xanchor="right",
            y=1.01,
            yanchor="bottom",
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=18, r=18, t=88, b=18),
        paper_bgcolor=C_BG,
        plot_bgcolor=C_BG,
        template="plotly_white",
        title={
            "text": title_text,
            "x": 0,
            "xanchor": "left",
            "font": {"size": 13},
        },
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#f0f0f0", zeroline=False)
    return fig


def note_figure(title: str, subtitle: str) -> go.Figure:
    """Return a simple placeholder figure for an empty filtered view."""
    fig = go.Figure()
    fig = base_layout(fig, title, subtitle)
    fig.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        text="No agencies match this view.",
        showarrow=False,
        font=dict(size=14, color=C_NEUTRAL),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=320)
    return fig


# -----------------------------------------------------------------------------
# sidebar
# -----------------------------------------------------------------------------
def build_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """Build sidebar filters and return the filtered dataframe."""
    st.sidebar.markdown(f"### {app_title.title()}")
    st.sidebar.markdown(
        "<span style='font-size:0.8rem;color:#888'>Field Education Review Dashboard</span>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Filters**")

    flag_view = st.sidebar.radio(
        "Flag status",
        ["All agencies", "Flagged only", "Not flagged only"],
        label_visibility="collapsed",
    )

    fit_min = float(df["placement_quality_score"].min())
    fit_max = float(df["placement_quality_score"].max())
    fit_range = st.sidebar.slider(
        "Placement Quality Score range",
        round(fit_min, 1),
        round(fit_max, 1),
        (round(fit_min, 1), round(fit_max, 1)),
        step=0.1,
    )

    min_n = int(df["response_count"].min())
    max_n = int(df["response_count"].max())
    response_floor = st.sidebar.slider(
        "Minimum responses",
        max(1, min_n),
        max_n,
        max(3, min_n),
    )

    trend_filter = st.sidebar.selectbox(
        "Placement quality trend",
        ["All trends", "Declining only", "Improving only", "Stable only"],
    )

    search = st.sidebar.text_input("Search agency name", "")

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Rerun python run_all.py to refresh the data. -Tomas"
    )

    filtered = df.copy()
    filtered = filtered[
        (filtered["placement_quality_score"] >= fit_range[0])
        & (filtered["placement_quality_score"] <= fit_range[1])
    ]
    filtered = filtered[filtered["response_count"] >= response_floor]

    if flag_view == "Flagged only":
        filtered = filtered[filtered["is_flagged"]]
    elif flag_view == "Not flagged only":
        filtered = filtered[~filtered["is_flagged"]]

    trend_map = {
        "Declining only": "declining",
        "Improving only": "improving",
        "Stable only": "stable",
    }
    if trend_filter != "All trends":
        filtered = filtered[filtered["fit_trend"] == trend_map[trend_filter]]

    if search.strip():
        filtered = filtered[
            filtered["agency_name_display"].str.contains(
                search.strip(),
                case=False,
                na=False,
            )
        ]

    return filtered.reset_index(drop=True)


# -----------------------------------------------------------------------------
# hero and KPIs
# -----------------------------------------------------------------------------
def render_hero(profiles: pd.DataFrame, trend_df: pd.DataFrame) -> None:
    """Render the hero block for the full placement view."""
    total = len(profiles)
    flagged = int(profiles["is_flagged"].sum())
    misaligned = int(profiles["is_misaligned"].sum())
    pct_flagged = round((flagged / total) * 100, 1)
    flagged_ratio_10 = round((flagged / total) * 10)

    latest_year = trend_df.sort_values("academic_year_start").iloc[-1]["academic_year"]
    latest_data = trend_df[trend_df["academic_year"] == latest_year]
    total_resp = latest_data["response_count"].sum()
    weighted_mean = (
        (latest_data["placement_quality_score"] * latest_data["response_count"]).sum()
        / total_resp
    )

    if weighted_mean < concern_fit_score:
        pqs_number = f"{weighted_mean:.2f}"
        pqs_color_class = "hero-number-concern"
        pqs_headline = "Program-wide score fell below the concern threshold for the first time."
        pqs_detail = (
            f"Mean Placement Quality Score dropped to {weighted_mean:.2f} in {latest_year}."
        )
    else:
        pqs_number = f"{weighted_mean:.2f}"
        pqs_color_class = ""
        pqs_headline = f"Program-wide score held above the concern threshold in {latest_year}."
        pqs_detail = (
            f"Mean Placement Quality Score was {weighted_mean:.2f}, above the {concern_fit_score:.1f} threshold."
        )

    if misaligned > 0:
        mis_headline = (
            f"Score-narrative mismatch in {misaligned} {'agency' if misaligned == 1 else 'agencies'}."
        )
        mis_detail = "These agencies met formal competency benchmarks but students reported some of the weakest experiences in the dataset."
    else:
        mis_headline = "No score-narrative mismatches this cycle."
        mis_detail = (
            "Agencies with strong benchmark scores also showed reasonable narrative feedback this cycle."
        )

    st.markdown(
        f"""
        <div class="hero-block">
            <div class="hero-eyebrow">This year's review</div>
            <div class="hero-title">What the {latest_year} evaluation data is showing</div>
            <div class="hero-row">
                <div>
                    <div class="hero-number hero-number-concern">{flagged}</div>
                    <div class="hero-headline">Nearly {flagged_ratio_10} in 10 agencies flagged for review.</div>
                    <div class="hero-detail">{flagged} of {total} active agencies met the concern threshold ({pct_flagged}%).</div>
                </div>
                <div>
                    <div class="hero-number {pqs_color_class}">{pqs_number}</div>
                    <div class="hero-headline">{pqs_headline}</div>
                    <div class="hero-detail">{pqs_detail}</div>
                </div>
                <div>
                    <div class="hero-number hero-number-caution">{misaligned}</div>
                    <div class="hero-headline">{mis_headline}</div>
                    <div class="hero-detail">{mis_detail}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(df: pd.DataFrame) -> None:
    """Render KPI tiles for the current filtered view."""
    total = len(df)
    flagged = int(df["is_flagged"].sum())
    misaligned = int(df["is_misaligned"].sum())
    pct_flagged = round((flagged / total) * 100, 1) if total else 0
    mean_fit = round(df["placement_quality_score"].mean(), 2)
    mean_rec = round(df["recommendation_rate_pct"].mean(), 1)

    filter_note = ""
    if "full_total" in df.attrs:
        full_total = df.attrs["full_total"]
        if full_total != total:
            dropped = full_total - total
            filter_note = (
                f" · Filter drops {dropped} of {full_total} agencies."
            )

    st.markdown(
        '<div class="quick-stats-label">At a glance - current filter view'
        + filter_note
        + "</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Flagged for review",
        flagged,
        f"{pct_flagged}% of current view",
        help="Agencies with 2 or more simultaneous concern signals. Start here.",
    )
    c2.metric("Agencies in view", total)
    c3.metric(
        "Mean Placement Quality Score",
        mean_fit,
        help=f"Concern threshold: {concern_fit_score:.1f}",
    )
    c4.metric("Mean recommendation rate", f"{mean_rec}%")
    c5.metric(
        "Score-narrative mismatch",
        misaligned,
        help="Agencies where benchmark scores look strong but student narrative is noticeably weaker.",
    )


# -----------------------------------------------------------------------------
# charts
# -----------------------------------------------------------------------------
THEME_ROOT_LABELS = {
    "administrative_overload": "Administrative overload",
    "direct_practice_opportunity": "Direct practice opportunity",
    "learning_environment": "Learning environment",
    "organizational_structure": "Organizational structure",
    "social_justice_alignment": "Social justice alignment",
    "strong_supervision": "Strong supervision",
}


def chart_flag_summary(df: pd.DataFrame) -> go.Figure:
    """Show the flagged share in the current filtered view."""
    flagged = int(df["is_flagged"].sum())
    not_flagged = int((~df["is_flagged"]).sum())
    total = max(flagged + not_flagged, 1)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[not_flagged / total * 100],
            y=["Current view"],
            orientation="h",
            marker_color=C_MUTED_BLUE,
            text=[f"<b>{not_flagged} agencies</b>"],
            textposition="inside",
            insidetextfont=dict(color=C_TEXT, size=13),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Bar(
            x=[flagged / total * 100],
            y=["Current view"],
            orientation="h",
            marker_color=C_CONCERN,
            text=[f"<b>{flagged} agencies</b>"],
            textposition="inside",
            insidetextfont=dict(color="#ffffff", size=13),
            showlegend=False,
        )
    )
    fig = base_layout(
        fig,
        f"Nearly {round(flagged / total * 10)} in 10 agencies in this view met the threshold for review",
        "An agency is flagged when two or more concern signals occur at the same time.",
    )
    fig.update_layout(barmode="stack", height=220)
    fig.update_xaxes(range=[0, 100], ticksuffix="%", title="Share of agencies")
    fig.update_yaxes(showticklabels=False, title="")
    return fig


def chart_fit_vs_recommend(df: pd.DataFrame) -> go.Figure:
    """Scatter of Placement Quality Score vs recommendation rate."""
    flagged = df[df["is_flagged"]].copy()
    not_flagged = df[~df["is_flagged"]].copy()

    fig = go.Figure()
    fig.add_shape(
        type="rect",
        x0=1.8,
        x1=concern_fit_score,
        y0=0,
        y1=concern_recommendation * 100,
        line=dict(width=0),
        fillcolor=C_SHADE,
        layer="below",
    )
    fig.add_trace(
        go.Scatter(
            x=not_flagged["placement_quality_score"],
            y=not_flagged["recommendation_rate_pct"],
            mode="markers",
            marker=dict(color=C_MUTED_BLUE, size=8, opacity=0.6),
            showlegend=False,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Placement Quality Score: %{x:.2f}<br>"
                "Recommendation rate: %{y:.1f}%<br>"
                "Responses: %{customdata[0]}<extra></extra>"
            ),
            text=not_flagged["agency_name_display"],
            customdata=not_flagged[["response_count"]],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=flagged["placement_quality_score"],
            y=flagged["recommendation_rate_pct"],
            mode="markers",
            marker=dict(color=C_CONCERN, size=10, opacity=0.85, symbol="triangle-up"),
            showlegend=False,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Placement Quality Score: %{x:.2f}<br>"
                "Recommendation rate: %{y:.1f}%<br>"
                "Responses: %{customdata[0]}<extra></extra>"
            ),
            text=flagged["agency_name_display"],
            customdata=flagged[["response_count"]],
        )
    )
    fig.add_hline(
        y=concern_recommendation * 100,
        line_color=C_SPINE,
        line_dash="dash",
        line_width=0.9,
    )
    fig.add_vline(
        x=concern_fit_score,
        line_color=C_SPINE,
        line_dash="dot",
        line_width=0.9,
    )
    fig.add_annotation(
        x=4.92,
        y=concern_recommendation * 100 + 1.5,
        text=f"{int(concern_recommendation * 100)}% threshold",
        showarrow=False,
        xanchor="right",
        font=dict(color=C_NEUTRAL, size=11),
    )
    fig.add_annotation(
        x=concern_fit_score + 0.03,
        y=2,
        text=f"{concern_fit_score:.1f} threshold",
        showarrow=False,
        xanchor="left",
        font=dict(color=C_NEUTRAL, size=11),
    )
    fig.add_annotation(
        x=1.95,
        y=60,
        text="<i>Both signals weak</i>",
        showarrow=False,
        xanchor="left",
        font=dict(color="#d9807a", size=11),
    )
    fig = base_layout(
        fig,
        "Flagged agencies cluster where placement quality and recommendation rates are both low",
        "Red triangles mark agencies needing review. Gray-blue dots show unflagged agencies. The shaded area marks where both measures fall below the concern thresholds.",
    )
    fig.update_xaxes(range=[1.8, 5.05], title="Placement Quality Score")
    fig.update_yaxes(range=[0, 104], title="Recommendation rate (%)")
    fig.update_layout(height=520)
    return fig


def chart_theme_summary(df: pd.DataFrame) -> go.Figure:
    """Show helpful vs least-helpful themes as a dumbbell chart."""
    rows = []
    for root in theme_dictionary:
        helpful_col = f"{root}_helpful_pct"
        least_col = f"{root}_least_pct"
        helpful_mean = round(df[helpful_col].mean(), 1)
        least_mean = round(df[least_col].mean(), 1)
        rows.append(
            {
                "theme": THEME_ROOT_LABELS[root],
                "helpful": helpful_mean,
                "least": least_mean,
                "gap": abs(helpful_mean - least_mean),
            }
        )

    theme_df = pd.DataFrame(rows).sort_values("gap", ascending=True)

    fig = go.Figure()
    for _, row in theme_df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["helpful"], row["least"]],
                y=[row["theme"], row["theme"]],
                mode="lines",
                line=dict(color=C_SPINE, width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=theme_df["helpful"],
            y=theme_df["theme"],
            mode="markers",
            marker=dict(color=C_POSITIVE, size=11),
            showlegend=False,
            hovertemplate="<b>%{y}</b><br>Helpful: %{x:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=theme_df["least"],
            y=theme_df["theme"],
            mode="markers",
            marker=dict(color=C_CONCERN, size=11),
            showlegend=False,
            hovertemplate="<b>%{y}</b><br>Least helpful: %{x:.1f}%<extra></extra>",
        )
    )
    fig = base_layout(
        fig,
        "Strong placements rely on supervision and real practice; weak ones suffer from paperwork and limited client contact",
        f"<span style='color:{C_POSITIVE}'><b>Blue</b></span> = appeared more often in most-helpful comments. "
        f"<span style='color:{C_CONCERN}'><b>Red</b></span> = appeared more often in least-helpful comments.",
    )
    fig.update_layout(showlegend=False, height=420)
    fig.update_xaxes(
        title="Mean % of agency responses tagged with this theme",
        range=[0, None],
    )
    return fig


def chart_lowest_fit_scores(df: pd.DataFrame) -> go.Figure:
    """Show the lowest Placement Quality Scores among flagged agencies in view."""
    flagged = df[df["is_flagged"]].copy()
    if flagged.empty:
        return note_figure(
            "No flagged agencies in the current view",
            "Adjust the sidebar filters to see the lowest-scoring flagged agencies.",
        )

    bottom = flagged.sort_values("placement_quality_score", ascending=True).head(10).copy()
    bottom["label"] = (
        bottom["agency_name_display"].apply(
            lambda s: s if len(s) <= 40 else s[:39].rstrip() + "…"
        )
        + "  (n="
        + bottom["response_count"].astype(int).astype(str)
        + ")"
    )

    fig = go.Figure(
        go.Bar(
            x=bottom["placement_quality_score"],
            y=bottom["label"],
            orientation="h",
            marker_color=C_CONCERN,
            text=bottom["placement_quality_score"].round(2),
            textposition="outside",
            texttemplate="%{text:.2f}",
        )
    )
    fig.add_vline(
        x=concern_fit_score,
        line_color=C_SPINE,
        line_dash="dot",
        line_width=0.9,
    )
    fig.add_annotation(
        x=concern_fit_score,
        y=-0.7,
        text=f"Concern threshold {concern_fit_score:.1f}",
        showarrow=False,
        xanchor="left",
        font=dict(color=C_NEUTRAL, size=10),
    )
    fig = base_layout(
        fig,
        "These flagged agencies have the lowest Placement Quality Scores in the current view",
        "Sorted lowest to highest. Use this as a quick starting point for review.",
    )
    fig.update_layout(showlegend=False, height=400)
    fig.update_xaxes(range=[0, 5.4], title="Placement Quality Score")
    fig.update_yaxes(autorange="reversed")
    return fig


def chart_trend_spotlight(trend_df: pd.DataFrame, profiles_df: pd.DataFrame) -> go.Figure:
    """Highlight the weakest flagged multi-year trajectories in the current view."""
    year_counts = (
        trend_df[trend_df["data_quality"] == "sufficient"]
        .groupby("agency_name")["academic_year"]
        .count()
    )
    sufficient = year_counts[year_counts >= 3].index
    trend_sufficient = trend_df[trend_df["agency_name"].isin(sufficient)].copy()

    spotlight_df = (
        profiles_df[
            profiles_df["is_flagged"] & profiles_df["agency_name"].isin(sufficient)
        ]
        .sort_values(
            ["concern_indicator_count", "placement_quality_score"],
            ascending=[False, True],
        )
        .head(3)
    )

    if spotlight_df.empty:
        return note_figure(
            "No multi-year spotlight agencies in the current view",
            "Adjust the sidebar filters to see flagged agencies with at least three years of sufficient trend data.",
        )

    spotlight_names = set(spotlight_df["agency_name"])
    fig = go.Figure()

    for agency_name, agency_df in trend_sufficient.groupby("agency_name"):
        if agency_name in spotlight_names:
            continue
        agency_sorted = agency_df.sort_values("academic_year_start")
        fig.add_trace(
            go.Scatter(
                x=agency_sorted["academic_year"],
                y=agency_sorted["placement_quality_score"],
                mode="lines",
                line=dict(color=C_MUTED_BLUE, width=0.8),
                opacity=0.35,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    yearly_mean = (
        trend_sufficient.groupby("academic_year")
        .apply(
            lambda g: (
                g["placement_quality_score"] * g["response_count"]
            ).sum()
            / g["response_count"].sum(),
            include_groups=False,
        )
        .reset_index()
        .rename(columns={0: "mean_score"})
        .merge(
            trend_sufficient[["academic_year", "academic_year_start"]].drop_duplicates(),
            on="academic_year",
        )
        .sort_values("academic_year_start")
    )

    fig.add_trace(
        go.Scatter(
            x=yearly_mean["academic_year"],
            y=yearly_mean["mean_score"].round(2),
            mode="lines",
            line=dict(color="#8d99ae", width=1.6, dash="dash"),
            showlegend=False,
            hovertemplate=(
                "<b>All-agency mean</b><br>"
                "Academic year: %{x}<br>"
                "Placement Quality Score: %{y:.2f}<extra></extra>"
            ),
        )
    )

    rose_shades = [C_CONCERN_DEEP, C_CONCERN, "#CC7A85"]
    label_items = []
    if not yearly_mean.empty:
        label_items.append(
            {
                "y": yearly_mean["mean_score"].iloc[-1],
                "text": "All-agency mean",
                "color": "#8d99ae",
                "bold": False,
            }
        )

    for i, (_, row) in enumerate(spotlight_df.iterrows()):
        agency_name = row["agency_name"]
        agency_df = trend_sufficient[
            trend_sufficient["agency_name"] == agency_name
        ].sort_values("academic_year_start")

        color = rose_shades[i % len(rose_shades)]
        display = row["agency_name_display"]
        short = display[:28] + "…" if len(display) > 28 else display

        fig.add_trace(
            go.Scatter(
                x=agency_df["academic_year"],
                y=agency_df["placement_quality_score"],
                mode="lines",
                line=dict(color=color, width=2.4),
                showlegend=False,
                customdata=agency_df[["response_count"]].values,
                hovertemplate=(
                    f"<b>{display}</b><br>"
                    "Academic year: %{x}<br>"
                    "Placement Quality Score: %{y:.2f}<br>"
                    "Responses: %{customdata[0]}<extra></extra>"
                ),
            )
        )

        label_items.append(
            {
                "y": agency_df["placement_quality_score"].iloc[-1],
                "text": short,
                "color": color,
                "bold": True,
            }
        )

    label_items.sort(key=lambda d: -d["y"])
    min_gap = 0.22
    placed_y = None
    for item in label_items:
        y_placed = item["y"] if placed_y is None else min(item["y"], placed_y - min_gap)
        placed_y = y_placed
        fig.add_annotation(
            x=1.01,
            y=y_placed,
            xref="paper",
            yref="y",
            text=item["text"],
            xanchor="left",
            showarrow=False,
            font=dict(
                color=item["color"],
                size=10,
                family="Arial Black" if item["bold"] else None,
            ),
        )

    fig.add_hline(
        y=concern_fit_score,
        line_color=C_SPINE,
        line_dash="dot",
        line_width=0.9,
    )
    fig.add_annotation(
        x=0,
        y=concern_fit_score + 0.05,
        xref="paper",
        text=f"Concern threshold  {concern_fit_score:.1f}",
        showarrow=False,
        xanchor="left",
        font=dict(color=C_NEUTRAL, size=10),
    )

    total_count = trend_sufficient["agency_name"].nunique()
    flagged_count = len(spotlight_names)

    fig = base_layout(
        fig,
        "Trend tracking exposes agencies that have stayed weak over time",
        f"Gray lines show all agencies. Highlighted agencies are flagged sites with the weakest recent multi-year trajectories. {flagged_count} of {total_count} agencies with sufficient multi-year data are highlighted here.",
    )
    fig.update_xaxes(title="Academic year", tickangle=40)
    fig.update_yaxes(range=[1.3, 5.3], title="Placement Quality Score")
    fig.update_layout(height=540, margin=dict(r=190))
    return fig


def summarize_program_trends(
    trend_df: pd.DataFrame,
    filtered_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize yearly trends for the agencies in the current filtered view."""
    names = filtered_profiles["agency_name"].unique().tolist()
    subset = trend_df[trend_df["agency_name"].isin(names)].copy()

    rows = []
    for academic_year, group in subset.groupby("academic_year"):
        total_responses = group["response_count"].sum()
        rows.append(
            {
                "academic_year": academic_year,
                "academic_year_start": group["academic_year_start"].iloc[0],
                "agencies": int(group["agency_name"].nunique()),
                "evaluations": int(total_responses),
                "flagged_agencies": int(group["is_flagged"].sum()),
                "mean_pqs": round(
                    (
                        group["placement_quality_score"] * group["response_count"]
                    ).sum()
                    / total_responses,
                    2,
                ),
                "mean_recommendation_rate_pct": round(
                    (
                        group["recommendation_rate_pct"] * group["response_count"]
                    ).sum()
                    / total_responses,
                    1,
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("academic_year_start")


def chart_program_trend(summary_df: pd.DataFrame, metric: str) -> go.Figure:
    """Show one trend over time for the agencies in the current view."""
    if summary_df.empty:
        return note_figure(
            "Trend unavailable for the current view",
            "No trend data matches the current filters.",
        )

    metric_map = {
        "Mean Placement Quality Score": (
            "mean_pqs",
            "Placement Quality Score",
            [2.5, 5.0],
        ),
        "Mean recommendation rate": (
            "mean_recommendation_rate_pct",
            "Recommendation rate (%)",
            [0, 100],
        ),
        "Flagged agencies": ("flagged_agencies", "Flagged agencies", None),
        "Evaluation count": ("evaluations", "Evaluations submitted", None),
    }
    col, axis_label, y_range = metric_map[metric]

    fig = px.line(summary_df, x="academic_year", y=col)
    fig = base_layout(fig, f"{metric} - trend across agencies in the current view")
    fig.update_traces(line_color=C_POSITIVE, line_width=2.4)

    if metric == "Mean Placement Quality Score":
        fig.add_hline(
            y=concern_fit_score,
            line_color=C_SPINE,
            line_dash="dot",
            line_width=0.9,
        )
        last_row = summary_df.iloc[-1]
        if last_row[col] < concern_fit_score:
            fig.add_trace(
                go.Scatter(
                    x=[last_row["academic_year"]],
                    y=[last_row[col]],
                    mode="markers",
                    marker=dict(color=C_CONCERN, size=11),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_annotation(
                x=last_row["academic_year"],
                y=last_row[col],
                text=f"<b>{last_row[col]:.2f}</b><br>Below threshold",
                showarrow=False,
                xanchor="right",
                xshift=-10,
                font=dict(color=C_CONCERN, size=10),
                align="right",
            )
    elif metric == "Mean recommendation rate":
        fig.add_hline(
            y=concern_recommendation * 100,
            line_color=C_SPINE,
            line_dash="dash",
            line_width=0.9,
        )

    fig.update_xaxes(title="Academic year", tickangle=40)
    fig.update_yaxes(range=y_range, title=axis_label)
    fig.update_layout(height=380)
    return fig


# -----------------------------------------------------------------------------
# agency review helpers
# -----------------------------------------------------------------------------
def flag_badge(row: pd.Series) -> str:
    """Return the concern flag badge."""
    if row["is_flagged"]:
        return '<span class="badge-flag">Review recommended</span>'
    return '<span class="badge-ok">No flag</span>'


def misalignment_badge(row: pd.Series) -> str:
    """Return the misalignment badge."""
    if row["is_misaligned"]:
        return '<span class="badge-mismatch">Score-narrative mismatch</span>'
    return ""


def trend_badge(row: pd.Series) -> str:
    """Return a colored trend badge."""
    trend = str(row["fit_trend"]).lower()
    if trend == "declining":
        return f'<span style="color:{C_CONCERN};font-weight:600">▼ Declining</span>'
    if trend == "improving":
        return f'<span style="color:{C_POSITIVE};font-weight:600">▲ Improving</span>'
    return f'<span style="color:{C_NEUTRAL}">- Stable</span>'


def render_top_phrases(value: object, limit: int = 8, top_n: int = 3) -> None:
    """Render saved bigrams with the top phrases emphasized."""
    if not isinstance(value, str) or not value.strip():
        st.caption("No phrase data available.")
        return

    phrases = [
        part.split(":")[0].strip()
        for part in value.split("|")
        if ":" in part
    ]

    if not phrases:
        st.caption("No phrase data available.")
        return

    top = phrases[:top_n]
    rest = phrases[top_n:limit]

    html = (
        f"<span style='font-size:0.92rem;font-weight:600;color:{C_TITLE}'>{', '.join(top)}</span>"
        + (
            f"<span style='font-size:0.82rem;color:{C_NEUTRAL}'>, {', '.join(rest)}</span>"
            if rest
            else ""
        )
    )
    st.markdown(html, unsafe_allow_html=True)


def chart_agency_trend(
    agency_trend_df: pd.DataFrame,
    label: str,
    trend_label: str = "stable",
) -> go.Figure:
    """Dual-axis agency trend: Placement Quality Score plus recommendation rate."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    s = agency_trend_df.sort_values("academic_year_start")

    if len(s) >= 2:
        last_score = s["placement_quality_score"].iloc[-1]
        last_year = s["academic_year"].iloc[-1]
        diff = last_score - s["placement_quality_score"].iloc[0]

        if trend_label == "declining" or diff < -0.30:
            title = f"{label} - placement quality declined to {last_score:.2f} in {last_year}"
        elif trend_label == "improving" or diff > 0.30:
            title = f"{label} - placement quality improved to {last_score:.2f} in {last_year}"
        else:
            title = f"{label} - placement quality remained fairly stable at {last_score:.2f} in {last_year}"
    else:
        title = f"{label} - placement quality trend over time"

    fig.add_trace(
        go.Scatter(
            x=s["academic_year"],
            y=s["placement_quality_score"],
            mode="lines+markers",
            name="Placement Quality Score",
            line=dict(color=C_POSITIVE, width=3),
            marker=dict(color=C_POSITIVE, size=7),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Placement Quality Score: %{y:.2f}<extra></extra>"
            ),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(
            x=s["academic_year"],
            y=s["recommendation_rate_pct"],
            name="Recommendation rate (%)",
            marker_color=C_LIGHT_BLUE,
            opacity=0.65,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Recommendation rate: %{y:.1f}%<extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    fig = base_layout(
        fig,
        title,
        "Line = Placement Quality Score. Bars = recommendation rate. Dotted lines mark the concern thresholds.",
    )

    fig.add_hline(
        y=concern_fit_score,
        line_color=C_SPINE,
        line_dash="dot",
        line_width=0.9,
        secondary_y=False,
    )
    fig.add_hline(
        y=concern_recommendation * 100,
        line_color=C_SPINE,
        line_dash="dash",
        line_width=0.9,
        secondary_y=True,
    )

    last_row = s.iloc[-1]
    if last_row["placement_quality_score"] < concern_fit_score:
        fig.add_trace(
            go.Scatter(
                x=[last_row["academic_year"]],
                y=[last_row["placement_quality_score"]],
                mode="markers",
                marker=dict(color=C_CONCERN, size=10),
                showlegend=False,
                hoverinfo="skip",
            ),
            secondary_y=False,
        )

    fig.update_xaxes(title="Academic year", tickangle=40)
    fig.update_yaxes(
        range=[1, 5.4],
        title_text="Placement Quality Score",
        secondary_y=False,
    )
    fig.update_yaxes(
        range=[0, 110],
        title_text="Recommendation rate (%)",
        secondary_y=True,
    )
    fig.update_layout(height=430)
    return fig


# -----------------------------------------------------------------------------
# agency review
# -----------------------------------------------------------------------------
def render_agency_review(
    filtered: pd.DataFrame,
    text_df: pd.DataFrame,
    trend_df: pd.DataFrame,
) -> None:
    """Render the full agency review section."""
    st.markdown(
        '<div class="chapter-eyebrow">Part 4 of 4</div>'
        '<div class="chapter-heading">Open an agency\'s story</div>'
        '<div class="chapter-bridge">When you want to review a specific agency, open its full profile below. '
        "You will see scores and signals up top, then theme patterns, representative student phrases, "
        "a year-by-year trend, and example evaluation text.</div>",
        unsafe_allow_html=True,
    )

    selected_label = st.selectbox(
        "Select an agency",
        filtered["agency_name_display"].tolist(),
        label_visibility="collapsed",
    )
    row = filtered.loc[filtered["agency_name_display"] == selected_label].iloc[0]

    badge_html = flag_badge(row)
    mis_badge = misalignment_badge(row)
    t_badge = trend_badge(row)

    comp_val = (
        f"{row['mean_competency_score']:.1f}"
        if pd.notna(row["mean_competency_score"])
        else "-"
    )
    resp_len = (
        f"{int(row['mean_response_length'])} words"
        if pd.notna(row["mean_response_length"])
        else "-"
    )

    st.markdown(
        f"""
        <div class="detail-card">
            <div style="font-size:1.25rem;font-weight:700;margin-bottom:6px">{selected_label}</div>
            <div style="margin-bottom:10px">{badge_html} &nbsp; {mis_badge if mis_badge else ""} &nbsp; {t_badge}</div>
            <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:12px">
                <div>
                    <div class="detail-label">Responses</div>
                    <div class="detail-value">{int(row['response_count'])}</div>
                </div>
                <div>
                    <div class="detail-label">Data quality</div>
                    <div class="detail-value">{str(row['data_quality']).title()}</div>
                </div>
                <div>
                    <div class="detail-label">Concern signals</div>
                    <div class="{'detail-value-concern' if row['concern_indicator_count'] >= 2 else 'detail-value'}">{int(row['concern_indicator_count'])}/5</div>
                </div>
                <div>
                    <div class="detail-label">Mean competency score</div>
                    <div class="detail-value">{comp_val}</div>
                </div>
                <div>
                    <div class="detail-label">Mean response length</div>
                    <div class="detail-value">{resp_len}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Placement Quality Score", f"{row['placement_quality_score']:.2f}")
    c2.metric("Recommendation rate", f"{row['recommendation_rate_pct']:.1f}%")
    c3.metric("Sentiment score", f"{row['overall_sentiment_score']:.3f}")

    if row["is_misaligned"]:
        st.markdown(
            """
            <div class="warn-box">
            <strong>Score-narrative mismatch detected.</strong> Program-level competency benchmarks appear strong for this cohort,
            but student open-ended responses describe a noticeably weaker placement experience. Formal scores and lived experience are
            telling different stories, so both deserve attention before drawing conclusions about this agency.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("**Theme profile**")
    theme_table = pd.DataFrame(
        {
            "Theme": [theme_labels[col] for col in theme_labels],
            "% of responses": [
                round(row[col], 1) if pd.notna(row[col]) else None
                for col in theme_labels
            ],
        }
    ).sort_values("% of responses", ascending=False)
    st.dataframe(
        theme_table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "% of responses": st.column_config.ProgressColumn(
                "% of responses",
                min_value=0,
                max_value=100,
                format="%.1f",
            )
        },
    )

    p1, p2 = st.columns(2)
    with p1:
        st.markdown("**Most-helpful phrases**")
        render_top_phrases(row["most_helpful_top_bigrams"])
    with p2:
        st.markdown("**Least-helpful phrases**")
        render_top_phrases(row["least_helpful_top_bigrams"])

    agency_trend = trend_df[trend_df["agency_name"] == row["agency_name"]].copy()
    st.plotly_chart(
        chart_agency_trend(
            agency_trend,
            selected_label,
            trend_label=str(row["fit_trend"]).lower(),
        ),
        use_container_width=True,
    )

    trend_table = agency_trend[
        [
            "academic_year",
            "response_count",
            "placement_quality_score",
            "recommendation_rate_pct",
            "overall_sentiment_score",
            "concern_flag",
        ]
    ].rename(
        columns={
            "academic_year": "Year",
            "response_count": "Responses",
            "placement_quality_score": "Placement Quality Score",
            "recommendation_rate_pct": "Rec. rate (%)",
            "overall_sentiment_score": "Sentiment",
            "concern_flag": "Flag",
        }
    )

    def _flag_style(val: str) -> str:
        if str(val).lower() == "review recommended":
            return f"color: {C_CONCERN}; font-weight: 600"
        return f"color: {C_NEUTRAL}"

    st.dataframe(
        trend_table.style.map(_flag_style, subset=["Flag"]),
        hide_index=True,
        use_container_width=True,
    )

    matches = text_df[text_df["agency_name"] == row["agency_name"]]
    if not matches.empty:
        with st.expander("Show example evaluation responses"):
            st.dataframe(
                matches[
                    [
                        "academic_year",
                        "most_helpful",
                        "least_helpful",
                        "comments_supervisor",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
                height=380,
            )


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main() -> None:
    """Run the Streamlit app."""
    profiles = load_profiles()
    text_df = load_text_data()
    trend_df = load_trend_data()

    st.markdown(
        f"<h2 style='margin-bottom:2px;letter-spacing:-0.02em'>{app_title.title()}</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#888;font-size:0.88rem;margin-top:0'>Field Education Review · University of Montana School of Social Work</p>",
        unsafe_allow_html=True,
    )

    render_hero(profiles, trend_df)

    filtered = build_sidebar(profiles)
    if filtered.empty:
        st.warning("No agencies match the current filters. Adjust the sidebar and try again.")
        return

    filtered.attrs["full_total"] = len(profiles)
    render_kpis(filtered)

    st.markdown(
        '<div class="chapter-eyebrow">Part 1 of 4</div>'
        '<div class="chapter-heading">Where concern clusters now</div>'
        '<div class="chapter-bridge">Start with the scatter view below. It shows which agencies have both a low placement quality score '
        "and a low recommendation rate. Red triangles are the flagged agencies. The blush zone marks where both signals "
        "fall below threshold at the same time.</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(chart_fit_vs_recommend(filtered), use_container_width=True)

    st.markdown(
        '<div class="chapter-eyebrow">Part 2 of 4</div>'
        '<div class="chapter-heading">Who has stayed weak over time</div>'
        '<div class="chapter-bridge">A low score in one year can happen for many reasons. The view below highlights the flagged sites '
        "whose placement quality has stayed low or dropped recently across multiple years.</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(chart_trend_spotlight(trend_df, filtered), use_container_width=True)

    st.markdown(
        '<div class="chapter-eyebrow">Part 3 of 4</div>'
        '<div class="chapter-heading">Supporting context</div>'
        '<div class="chapter-bridge">These views add supporting context to the signals above. The flag summary and score rankings help '
        "prioritize which flagged agency to look at first. The theme chart shows what students describe as helpful and what they find least helpful.</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    c1.plotly_chart(chart_flag_summary(filtered), use_container_width=True)
    c2.plotly_chart(chart_lowest_fit_scores(filtered), use_container_width=True)

    c3, c4 = st.columns([1.1, 0.9])
    c3.plotly_chart(chart_theme_summary(filtered), use_container_width=True)

    with c4:
        program_summary = summarize_program_trends(trend_df, filtered)
        trend_metric = st.radio(
            "Program-wide metric",
            [
                "Mean Placement Quality Score",
                "Mean recommendation rate",
                "Flagged agencies",
                "Evaluation count",
            ],
            horizontal=False,
        )
        st.plotly_chart(
            chart_program_trend(program_summary, trend_metric),
            use_container_width=True,
        )

    with st.expander("Show agency summary table"):
        st.dataframe(
            filtered[
                [
                    "agency_name_display",
                    "response_count",
                    "placement_quality_score",
                    "recommendation_rate_pct",
                    "overall_sentiment_score",
                    "mean_competency_score",
                    "fit_trend",
                    "concern_indicator_count",
                    "concern_flag",
                    "misalignment_flag",
                ]
            ].rename(
                columns={
                    "agency_name_display": "Agency",
                    "placement_quality_score": "Placement Quality Score",
                    "concern_flag": "Flag",
                    "misalignment_flag": "Misalignment",
                    "overall_sentiment_score": "Sentiment",
                    "recommendation_rate_pct": "Rec. rate (%)",
                    "response_count": "Responses",
                    "concern_indicator_count": "Signals",
                    "mean_competency_score": "Mean comp. score",
                    "fit_trend": "Trend",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    render_agency_review(filtered, text_df, trend_df)


if __name__ == "__main__":
    main()
