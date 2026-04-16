"""Streamlit dashboard for practicum evaluation intelligence.

This app is the leadership-facing view of the pipeline output files. It does
not rerun the pipeline. It reads the saved agency profile and trend tables so
field education staff can review agencies, compare patterns, and look more
closely at sites that may need attention.

Run from the project root:
    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config import (
    agency_profiles_file,
    agency_trends_file,
    app_title,
    evaluations_text_file,
)
from constants import detail_metrics, theme_labels

# ------------------------------------------------------------------------------
# Color palette - matches create_visualizations.py exactly
# ------------------------------------------------------------------------------

C_CONCERN = "#c0392b"     # strong red - flagged / problematic
C_CAUTION = "#e67e22"     # orange - borderline / misalignment
C_NEUTRAL = "#7f8c8d"     # gray - reference lines, muted text
C_POSITIVE = "#2471a3"    # steel blue - good / no flag
C_LIGHT = "#aad2e4"       # light blue - volume bars
C_BG = "#ffffff"
C_SURFACE = "#f8f9fa"
C_BORDER = "#e5e7eb"

# ------------------------------------------------------------------------------
# Page config and CSS
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title=app_title,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
        /* page breathing room */
        .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }}

        /* KPI metric tiles */
        div[data-testid="metric-container"] {{
            background-color: {C_SURFACE};
            border: 1px solid {C_BORDER};
            border-radius: 10px;
            padding: 16px 20px;
        }}

        /* section headers */
        .section-header {{
            font-size: 1.05rem;
            font-weight: 700;
            color: #1a1a1a;
            margin-top: 1.5rem;
            margin-bottom: 0.15rem;
            letter-spacing: -0.01em;
        }}
        .section-sub {{
            font-size: 0.85rem;
            color: {C_NEUTRAL};
            margin-bottom: 0.9rem;
        }}

        /* flag badges */
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

        /* note / info box */
        .note-box {{
            background: {C_SURFACE};
            border-left: 4px solid {C_POSITIVE};
            padding: 0.75rem 1rem;
            border-radius: 0 6px 6px 0;
            font-size: 0.88rem;
            color: #444;
            margin-bottom: 1rem;
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

        /* agency detail card */
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
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------


@st.cache_data
def load_profiles() -> pd.DataFrame:
    """Load the agency profiles table. Validates required columns on load."""
    if not agency_profiles_file.exists():
        st.error(
            f"Could not find agency_profiles.csv at {agency_profiles_file}. "
            "Run the pipeline first: python pipeline.py"
        )
        st.stop()

    df = pd.read_csv(agency_profiles_file)

    required = {"agency_name", "composite_fit_score", "concern_flag", "recommendation_rate", "response_count"}
    missing = sorted(required.difference(df.columns))
    if missing:
        st.error(f"Profile file is missing columns: {', '.join(missing)}. Rerun the pipeline.")
        st.stop()

    numeric_cols = [
        "composite_fit_score", "concern_indicator_count", "overall_sentiment_score",
        "recommendation_rate", "response_count", "mean_competency_score",
        *detail_metrics.keys(),
        *[col for col in theme_labels if col in df.columns],
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # normalize recommendation rate to 0-1 internally, expose as pct for display
    if "recommendation_rate" in df.columns and df["recommendation_rate"].dropna().max() <= 1.0:
        df["recommendation_rate_pct"] = df["recommendation_rate"] * 100
    else:
        df["recommendation_rate_pct"] = df["recommendation_rate"]
        df["recommendation_rate"] = df["recommendation_rate"] / 100

    if "agency_name_display" not in df.columns:
        df["agency_name_display"] = df["agency_name"].astype(str).str.title()

    df["concern_flag"] = df.get("concern_flag", pd.Series("no flag", index=df.index)).fillna("no flag")
    df["misalignment_flag"] = df.get("misalignment_flag", pd.Series("no data", index=df.index)).fillna("no data")
    df["data_quality"] = df.get("data_quality", pd.Series("unknown", index=df.index)).fillna("unknown")
    df["fit_trend"] = df.get("fit_trend", pd.Series("stable", index=df.index)).fillna("stable")
    df["is_flagged"] = df["concern_flag"].str.lower().eq("review recommended")
    df["is_misaligned"] = df["misalignment_flag"].str.lower().eq("score-narrative mismatch")

    return df.sort_values(
        ["is_flagged", "composite_fit_score", "agency_name_display"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


@st.cache_data
def load_text_data() -> pd.DataFrame | None:
    """Load the optional evaluation text table if available."""
    if not evaluations_text_file.exists():
        return None
    return pd.read_csv(evaluations_text_file)


@st.cache_data
def load_trend_data() -> pd.DataFrame | None:
    """Load the yearly trend table if available."""
    if not agency_trends_file.exists():
        return None

    df = pd.read_csv(agency_trends_file)
    numeric_cols = [
        "academic_year_start", "composite_fit_score", "concern_indicator_count",
        "overall_sentiment_score", "recommendation_rate", "response_count",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "recommendation_rate" in df.columns and df["recommendation_rate"].dropna().max() <= 1.0:
        df["recommendation_rate_pct"] = df["recommendation_rate"] * 100
    else:
        df["recommendation_rate_pct"] = df["recommendation_rate"]
        df["recommendation_rate"] = df["recommendation_rate"] / 100

    if "agency_name_display" not in df.columns:
        df["agency_name_display"] = df["agency_name"].astype(str).str.title()

    df["concern_flag"] = df.get("concern_flag", pd.Series("no flag", index=df.index)).fillna("no flag")
    df["is_flagged"] = df["concern_flag"].str.lower().eq("review recommended")
    return df.sort_values(["academic_year_start", "agency_name_display"])


# ------------------------------------------------------------------------------
# Shared Plotly layout
# ------------------------------------------------------------------------------


def base_layout(fig: go.Figure, title: str, subtitle: str | None = None) -> go.Figure:
    """Apply one shared Plotly layout so every chart looks consistent."""
    title_text = f"<b>{title}</b>"
    if subtitle:
        title_text += f"<br><span style='font-size:11px;color:{C_NEUTRAL}'>{subtitle}</span>"
    fig.update_layout(
        font=dict(family="sans-serif", size=12, color="#1a1a1a"),
        legend=dict(orientation="h", x=1, xanchor="right", y=1.01, yanchor="bottom", bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=16, r=16, t=70, b=16),
        paper_bgcolor=C_BG,
        plot_bgcolor=C_BG,
        template="plotly_white",
        title={"text": title_text, "x": 0, "xanchor": "left", "font": {"size": 13}},
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#f0f0f0", zeroline=False)
    return fig


# ------------------------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------------------------


def build_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """Build sidebar filters and return the filtered dataframe."""
    st.sidebar.markdown(f"### {app_title.title()}")
    st.sidebar.markdown(
        "<span style='font-size:0.8rem;color:#888'>Field education leadership dashboard</span>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Filters**")

    flag_view = st.sidebar.radio(
        "Flag status",
        ["All agencies", "Flagged only", "Not flagged only"],
        label_visibility="collapsed",
    )

    fit_min = float(df["composite_fit_score"].min())
    fit_max = float(df["composite_fit_score"].max())
    fit_range = st.sidebar.slider(
        "Placement Quality Score range",
        round(fit_min, 1),
        round(fit_max, 1),
        (round(fit_min, 1), round(fit_max, 1)),
        step=0.1,
    )

    min_n = int(df["response_count"].min())
    max_n = int(df["response_count"].max())
    response_floor = st.sidebar.slider("Minimum responses", max(1, min_n), max_n, max(3, min_n))

    trend_filter = st.sidebar.selectbox(
        "Fit trend",
        ["All trends", "Declining only", "Improving only", "Stable only"],
    )

    search = st.sidebar.text_input("Search agency name", "")

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "This app reads saved pipeline outputs. "
        "Rerun pipeline.py to refresh the data."
    )

    filtered = df.copy()
    filtered = filtered[
        (filtered["composite_fit_score"] >= fit_range[0])
        & (filtered["composite_fit_score"] <= fit_range[1])
    ]
    filtered = filtered[filtered["response_count"] >= response_floor]

    if flag_view == "Flagged only":
        filtered = filtered[filtered["is_flagged"]]
    elif flag_view == "Not flagged only":
        filtered = filtered[~filtered["is_flagged"]]

    if trend_filter != "All trends" and "fit_trend" in filtered.columns:
        trend_map = {
            "Declining only": "declining",
            "Improving only": "improving",
            "Stable only": "stable",
        }
        filtered = filtered[filtered["fit_trend"] == trend_map[trend_filter]]

    if search.strip():
        filtered = filtered[
            filtered["agency_name_display"].str.contains(search.strip(), case=False, na=False)
        ]

    return filtered.reset_index(drop=True)


# ------------------------------------------------------------------------------
# KPI tiles
# ------------------------------------------------------------------------------


def render_kpis(df: pd.DataFrame) -> None:
    """Render the four top summary metrics for the current filtered view."""
    total = len(df)
    flagged = int(df["is_flagged"].sum())
    misaligned = int(df["is_misaligned"].sum()) if "is_misaligned" in df.columns else 0
    pct_flagged = round((flagged / total) * 100, 1) if total else 0
    mean_fit = round(df["composite_fit_score"].mean(), 2) if total else 0
    mean_rec = round(df["recommendation_rate_pct"].mean(), 1) if total else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Agencies in view", total)
    c2.metric("Flagged for review", flagged, f"{pct_flagged}% of view")
    c3.metric("Score-narrative mismatch", misaligned)
    c4.metric("Mean Placement Quality Score", mean_fit)
    c5.metric("Mean recommendation rate", f"{mean_rec}%")

    st.markdown(
        """<div class="note-box">
        This dashboard supports leadership review — it is not a final decision tool.
        Agencies are best interpreted alongside response count, qualitative themes,
        program context, and field team knowledge.
        </div>""",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------------------
# Portfolio charts
# ------------------------------------------------------------------------------


def chart_flag_summary(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar: flagged vs not flagged in the current view."""
    summary = pd.DataFrame({
        "Status": ["No flag", "Review recommended"],
        "Count": [int((~df["is_flagged"]).sum()), int(df["is_flagged"].sum())],
    })
    fig = px.bar(
        summary,
        color="Status",
        color_discrete_map={"No flag": C_POSITIVE, "Review recommended": C_CONCERN},
        orientation="h",
        text="Count",
        x="Count",
        y="Status",
    )
    fig = base_layout(fig, "Flag status in current view")
    fig.update_layout(showlegend=False)
    fig.update_traces(cliponaxis=False, textposition="outside")
    return fig


def chart_fit_vs_recommend(df: pd.DataFrame) -> go.Figure:
    """Scatter: placement quality vs recommendation rate, colored by flag."""
    fig = px.scatter(
        df,
        color="concern_flag",
        color_discrete_map={"no flag": C_POSITIVE, "review recommended": C_CONCERN},
        hover_data={"response_count": True, "overall_sentiment_score": ":.3f"},
        hover_name="agency_name_display",
        symbol="concern_flag",
        symbol_map={"no flag": "circle", "review recommended": "triangle-up"},
        x="composite_fit_score",
        y="recommendation_rate_pct",
    )
    fig.add_hline(y=70, line_color=C_NEUTRAL, line_dash="dash", line_width=1)
    fig.add_vline(x=3.5, line_color=C_NEUTRAL, line_dash="dot", line_width=1)
    fig = base_layout(
        fig,
        "Placement quality vs recommendation rate",
        "Flagged agencies cluster toward the lower-left. Reference lines show current cut points.",
    )
    fig.update_xaxes(range=[1.8, 5.1], title="Placement Quality Score")
    fig.update_yaxes(range=[0, 108], title="Recommendation rate (%)")
    return fig


def chart_theme_heatmap(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar: average theme rates sorted by most frequent."""
    available = [col for col in theme_labels if col in df.columns]
    if not available:
        return go.Figure()

    theme_data = pd.DataFrame({
        "Theme": [theme_labels[col] for col in available],
        "Rate": [round(df[col].mean(), 1) for col in available],
    }).sort_values("Rate", ascending=True)

    fig = px.bar(
        theme_data,
        orientation="h",
        text="Rate",
        x="Rate",
        y="Theme",
    )
    fig = base_layout(
        fig,
        "Average theme rate across agencies in current view",
        "Shows how often each theme appears across the filtered agency set",
    )
    fig.update_layout(showlegend=False)
    fig.update_traces(marker_color=C_POSITIVE, textposition="outside", texttemplate="%{text:.1f}%")
    fig.update_xaxes(title="Average % of responses tagged")
    return fig


def chart_bottom_ten(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar: ten lowest fit scores in the current filtered view."""
    bottom = (
        df.sort_values("composite_fit_score", ascending=True)
        .head(10)
        .copy()
    )
    bottom["label"] = bottom["agency_name_display"] + "  (n=" + bottom["response_count"].astype(str) + ")"
    colors = [C_CONCERN if row["is_flagged"] else C_CAUTION for _, row in bottom.iterrows()]

    fig = go.Figure(go.Bar(
        orientation="h",
        text=bottom["composite_fit_score"].round(2),
        textposition="outside",
        texttemplate="%{text:.2f}",
        x=bottom["composite_fit_score"],
        y=bottom["label"],
        marker_color=colors,
    ))
    fig.add_vline(x=3.5, line_color=C_NEUTRAL, line_dash="dot", line_width=1)
    fig = base_layout(fig, "Ten lowest Placement Quality Scores in current view")
    fig.update_layout(showlegend=False)
    fig.update_xaxes(range=[0, 5.4], title="Placement Quality Score")
    fig.update_yaxes(autorange="reversed")
    return fig


# ------------------------------------------------------------------------------
# Interactive spotlight trend chart (Plotly version of fig_15)
# ------------------------------------------------------------------------------


def chart_trend_spotlight(trend_df: pd.DataFrame, profiles_df: pd.DataFrame) -> go.Figure:
    """Interactive gray-out trend chart: all agencies in gray, flagged ones in red.

    This is the Plotly version of the matplotlib fig_15 so it is interactive
    in the dashboard - users can hover over any line to see the agency name
    and scores for that year.
    """
    # agencies with at least 3 years of trend data
    year_counts = (
        trend_df[trend_df["data_quality"] == "sufficient"]
        .groupby("agency_name")["academic_year"]
        .count()
    )
    sufficient = year_counts[year_counts >= 3].index
    trend_sufficient = trend_df[trend_df["agency_name"].isin(sufficient)].copy()

    # top 6 flagged agencies
    display_col = "agency_name_display" if "agency_name_display" in profiles_df.columns else "agency_name"
    spotlight_df = (
        profiles_df[
            profiles_df["concern_flag"].str.lower().eq("review recommended")
            & profiles_df["agency_name"].isin(sufficient)
        ]
        .sort_values(["concern_indicator_count", "composite_fit_score"], ascending=[False, True])
        .head(6)
    )
    spotlight_names = set(spotlight_df["agency_name"])

    fig = go.Figure()

    # layer 1: all background agencies in gray
    for agency_name, agency_df in trend_sufficient.groupby("agency_name"):
        if agency_name in spotlight_names:
            continue
        agency_sorted = agency_df.sort_values("academic_year_start")
        display = agency_sorted["agency_name_display"].iloc[0]
        fig.add_trace(go.Scatter(
            hoverinfo="skip",
            line=dict(color=C_NEUTRAL, width=0.8),
            mode="lines",
            name=display,
            opacity=0.18,
            showlegend=False,
            x=agency_sorted["academic_year"],
            y=agency_sorted["composite_fit_score"],
        ))

    # layer 2: program mean
    yearly_mean = (
        trend_sufficient.groupby("academic_year")
        .apply(
            lambda g: (g["composite_fit_score"] * g["response_count"]).sum()
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
    fig.add_trace(go.Scatter(
        line=dict(color="#2c3e50", width=2.0, dash="dash"),
        mode="lines",
        name="Program mean",
        x=yearly_mean["academic_year"],
        y=yearly_mean["mean_score"].round(2),
    ))

    # layer 3: spotlight agencies in red shades
    red_shades = ["#c0392b", "#e74c3c", "#a93226", "#cb4335", "#b03a2e", "#d98880"]
    for i, (_, srow) in enumerate(spotlight_df.iterrows()):
        agency_name = srow["agency_name"]
        agency_df = trend_sufficient[trend_sufficient["agency_name"] == agency_name].sort_values("academic_year_start")
        color = red_shades[i % len(red_shades)]
        display = srow[display_col]
        short = display[:28] + "…" if len(display) > 28 else display

        fig.add_trace(go.Scatter(
            customdata=agency_df[["response_count", "concern_flag"]].values,
            hovertemplate=(
                f"<b>{display}</b><br>"
                "Year: %{x}<br>"
                "Score: %{y:.2f}<br>"
                "Responses: %{customdata[0]}<br>"
                "<extra></extra>"
            ),
            line=dict(color=color, width=2.5),
            marker=dict(color=color, size=5),
            mode="lines+markers",
            name=short,
            x=agency_df["academic_year"],
            y=agency_df["composite_fit_score"],
        ))

    # low-fit reference line
    fig.add_hline(y=3.5, line_color=C_CAUTION, line_dash="dot", line_width=1, opacity=0.5)

    total_count = trend_sufficient["agency_name"].nunique()
    flagged_count = len(spotlight_names)

    fig = base_layout(
        fig,
        f"Agency placement quality trends — {flagged_count} flagged agencies highlighted against {total_count} total",
        "Gray lines show all agencies with 3+ years of data. Hover any highlighted line to see details.",
    )
    fig.update_xaxes(title="Academic year", tickangle=40)
    fig.update_yaxes(range=[1.5, 5.3], title="Placement Quality Score (1-5)")
    fig.update_layout(
        height=500,
        legend=dict(orientation="v", x=1.01, xanchor="left", y=1, yanchor="top"),
    )
    return fig


# ------------------------------------------------------------------------------
# Portfolio trend summary
# ------------------------------------------------------------------------------


def summarize_portfolio_trends(trend_df: pd.DataFrame, filtered_profiles: pd.DataFrame) -> pd.DataFrame:
    """Summarize yearly trends for the agencies in the current filtered view."""
    names = filtered_profiles["agency_name"].unique().tolist()
    subset = trend_df[trend_df["agency_name"].isin(names)].copy()
    if subset.empty:
        return subset

    rows = []
    for academic_year, group in subset.groupby("academic_year"):
        total_responses = group["response_count"].sum()
        rows.append({
            "academic_year": academic_year,
            "academic_year_start": group["academic_year_start"].iloc[0],
            "agencies": int(group["agency_name"].nunique()),
            "evaluations": int(total_responses),
            "flagged_agencies": int(group["is_flagged"].sum()),
            "mean_fit_score": round(
                (group["composite_fit_score"] * group["response_count"]).sum() / total_responses, 2
            ),
            "mean_recommendation_rate_pct": round(
                (group["recommendation_rate_pct"] * group["response_count"]).sum() / total_responses, 1
            ),
            "mean_sentiment_score": round(
                (group["overall_sentiment_score"] * group["response_count"]).sum() / total_responses, 3
            ),
        })
    return pd.DataFrame(rows).sort_values("academic_year_start")


def chart_portfolio_trend(summary_df: pd.DataFrame, metric: str) -> go.Figure:
    """Line chart for one portfolio-wide metric over time."""
    metric_map = {
        "Mean fit score": ("mean_fit_score", "Mean fit score", [2.5, 5.0]),
        "Mean recommendation rate": ("mean_recommendation_rate_pct", "Recommendation rate (%)", [0, 100]),
        "Flagged agencies": ("flagged_agencies", "Flagged agencies", None),
        "Evaluation count": ("evaluations", "Evaluations submitted", None),
    }
    col, axis_label, y_range = metric_map[metric]
    fig = px.line(summary_df, markers=True, text=col, x="academic_year", y=col)
    fig = base_layout(fig, f"Portfolio trend: {metric.lower()}")
    fig.update_traces(line_color=C_POSITIVE, line_width=2.5, marker_size=7, textposition="top center")

    if metric == "Mean fit score":
        fig.add_hline(y=3.5, line_color=C_NEUTRAL, line_dash="dot", line_width=1)
        fig.update_traces(texttemplate="%{text:.2f}")
    elif metric == "Mean recommendation rate":
        fig.add_hline(y=70, line_color=C_NEUTRAL, line_dash="dash", line_width=1)
        fig.update_traces(texttemplate="%{text:.1f}%")
    else:
        fig.update_traces(texttemplate="%{text:.0f}")

    fig.update_xaxes(title="Academic year", tickangle=40)
    fig.update_yaxes(range=y_range, title=axis_label)
    return fig


# ------------------------------------------------------------------------------
# Agency drill-down
# ------------------------------------------------------------------------------


def flag_badge(row: pd.Series) -> str:
    """Return HTML badge for the concern flag status."""
    if row.get("is_flagged", False):
        return '<span class="badge-flag">Review recommended</span>'
    return '<span class="badge-ok">No flag</span>'


def misalignment_badge(row: pd.Series) -> str:
    """Return HTML badge for the misalignment flag if present."""
    flag = str(row.get("misalignment_flag", "no data")).lower()
    if flag == "score-narrative mismatch":
        return '<span class="badge-mismatch">Score-narrative mismatch</span>'
    return ""


def trend_badge(row: pd.Series) -> str:
    """Return a colored trend indicator."""
    trend = str(row.get("fit_trend", "stable")).lower()
    if trend == "declining":
        return f'<span style="color:{C_CONCERN};font-weight:600">▼ Declining</span>'
    if trend == "improving":
        return f'<span style="color:{C_POSITIVE};font-weight:600">▲ Improving</span>'
    return f'<span style="color:{C_NEUTRAL}">— Stable</span>'


def summarize_top_phrases(value: object, limit: int = 8) -> str:
    """Turn a saved bigram summary string into a short readable list."""
    if not isinstance(value, str) or not value.strip():
        return "No phrase data available."
    phrases = [part.split(":")[0].strip() for part in value.split("|") if ":" in part]
    return ", ".join(phrases[:limit]) if phrases else "No phrase data available."


def chart_agency_trend(agency_trend_df: pd.DataFrame, label: str) -> go.Figure:
    """Dual-axis chart: fit score line + recommendation rate bars for one agency."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    agency_sorted = agency_trend_df.sort_values("academic_year_start")

    fig.add_trace(
        go.Scatter(
            line=dict(color=C_POSITIVE, width=3),
            mode="lines+markers+text",
            name="Placement Quality Score",
            text=[f"{v:.2f}" for v in agency_sorted["composite_fit_score"]],
            textposition="top center",
            x=agency_sorted["academic_year"],
            y=agency_sorted["composite_fit_score"],
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            marker_color=C_LIGHT,
            name="Recommendation rate (%)",
            opacity=0.7,
            x=agency_sorted["academic_year"],
            y=agency_sorted["recommendation_rate_pct"],
        ),
        secondary_y=True,
    )
    fig = base_layout(
        fig,
        f"Year-by-year view: {label}",
        "Line = Placement Quality Score (left axis). Bars = recommendation rate (right axis).",
    )
    fig.add_hline(y=3.5, line_color=C_NEUTRAL, line_dash="dot", secondary_y=False)
    fig.add_hline(y=70, line_color=C_NEUTRAL, line_dash="dash", secondary_y=True)
    fig.update_xaxes(title="Academic year", tickangle=40)
    fig.update_yaxes(range=[1, 5.4], title_text="Placement Quality Score", secondary_y=False)
    fig.update_yaxes(range=[0, 110], title_text="Recommendation rate (%)", secondary_y=True)
    return fig


def render_agency_detail(
    filtered: pd.DataFrame,
    text_df: pd.DataFrame | None,
    trend_df: pd.DataFrame | None,
    all_profiles: pd.DataFrame,
) -> None:
    """Agency drill-down: metrics, themes, phrases, trend chart, raw text."""
    st.markdown('<div class="section-header">Agency drill-down</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Select one agency to review its full profile, language patterns, and over-time trend.</div>',
        unsafe_allow_html=True,
    )

    options = filtered["agency_name_display"].tolist()
    selected_label = st.selectbox("Select an agency", options, label_visibility="collapsed")
    row = filtered.loc[filtered["agency_name_display"] == selected_label].iloc[0]

    # header row: name, badges, response count
    badge_html = flag_badge(row)
    mis_badge = misalignment_badge(row)
    t_badge = trend_badge(row)
    st.markdown(
        f"""
        <div class="detail-card">
            <div style="font-size:1.25rem;font-weight:700;margin-bottom:6px">{selected_label}</div>
            <div style="margin-bottom:10px">{badge_html} &nbsp; {mis_badge if mis_badge else ""} &nbsp; {t_badge}</div>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px">
                <div>
                    <div class="detail-label">Responses</div>
                    <div class="detail-value">{int(row['response_count'])}</div>
                </div>
                <div>
                    <div class="detail-label">Data quality</div>
                    <div class="detail-value">{row['data_quality'].title()}</div>
                </div>
                <div>
                    <div class="detail-label">Concern signals</div>
                    <div class="{'detail-value-concern' if row['concern_indicator_count'] >= 2 else 'detail-value'}">{int(row['concern_indicator_count'])}/5</div>
                </div>
                <div>
                    <div class="detail-label">Mean competency score</div>
                    <div class="detail-value">{f"{row['mean_competency_score']:.1f}" if 'mean_competency_score' in row and pd.notna(row['mean_competency_score']) else "—"}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # key metrics row
    col1, col2, col3 = st.columns(3)
    col1.metric("Placement Quality Score", f"{row['composite_fit_score']:.2f}")
    col2.metric("Recommendation rate", f"{row['recommendation_rate_pct']:.1f}%")
    col3.metric("Sentiment score", f"{row['overall_sentiment_score']:.3f}")

    # misalignment context box if flagged
    if row.get("is_misaligned", False):
        st.markdown(
            f"""<div class="warn-box">
            <strong>Score-narrative mismatch detected.</strong> Program competency scores for this
            agency's cohort appear strong, but open-ended responses contain lower sentiment and
            higher administrative overload signals than expected. This may warrant a closer look
            at whether the formal evaluation scores are reflecting actual student experience.
            </div>""",
            unsafe_allow_html=True,
        )

    # theme profile
    theme_cols = [col for col in theme_labels if col in filtered.columns]
    if theme_cols:
        st.markdown("**Theme profile**")
        theme_table = pd.DataFrame({
            "Theme": [theme_labels[col] for col in theme_cols],
            "% of responses": [round(row[col], 1) if pd.notna(row[col]) else None for col in theme_cols],
        }).sort_values("% of responses", ascending=False)
        st.dataframe(theme_table, hide_index=True, use_container_width=True)

    # top phrases
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("**Most-helpful phrases**")
        st.caption(summarize_top_phrases(row.get("most_helpful_top_bigrams")))
    with p2:
        st.markdown("**Least-helpful phrases**")
        st.caption(summarize_top_phrases(row.get("least_helpful_top_bigrams")))

    # over-time trend chart
    if trend_df is not None:
        agency_trend = trend_df[trend_df["agency_name"] == row["agency_name"]].copy()
        if len(agency_trend) > 0:
            st.plotly_chart(
                chart_agency_trend(agency_trend, selected_label),
                use_container_width=True,
            )
            trend_table = agency_trend[[
                "academic_year", "response_count", "composite_fit_score",
                "recommendation_rate_pct", "overall_sentiment_score", "concern_flag",
            ]].rename(columns={
                "academic_year": "Year",
                "response_count": "Responses",
                "composite_fit_score": "Fit score",
                "recommendation_rate_pct": "Rec. rate (%)",
                "overall_sentiment_score": "Sentiment",
                "concern_flag": "Flag",
            })
            st.dataframe(trend_table, hide_index=True, use_container_width=True)

    # example evaluation text
    if text_df is not None and "agency_name" in text_df.columns:
        matches = text_df[text_df["agency_name"] == row["agency_name"]]
        if not matches.empty:
            with st.expander("Show example evaluation responses"):
                preview_cols = [
                    col for col in ["academic_year", "most_helpful", "least_helpful", "comments_supervisor"]
                    if col in matches.columns
                ]
                st.dataframe(matches[preview_cols].head(5), hide_index=True, use_container_width=True)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------


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
        "<p style='color:#888;font-size:0.88rem;margin-top:0'>Field education leadership dashboard — University of Montana School of Social Work</p>",
        unsafe_allow_html=True,
    )

    filtered = build_sidebar(profiles)

    if filtered.empty:
        st.warning("No agencies match the current filters. Adjust the sidebar and try again.")
        return

    render_kpis(filtered)

    # portfolio overview
    st.markdown('<div class="section-header">Portfolio overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Charts below reflect the agencies in the current filtered view.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    c1.plotly_chart(chart_flag_summary(filtered), use_container_width=True)
    c2.plotly_chart(chart_fit_vs_recommend(filtered), use_container_width=True)

    c3, c4 = st.columns(2)
    c3.plotly_chart(chart_theme_heatmap(filtered), use_container_width=True)
    c4.plotly_chart(chart_bottom_ten(filtered), use_container_width=True)

    # agency summary table
    st.markdown("**Agency summary table**")
    table_cols = [col for col in [
        "agency_name_display", "response_count", "composite_fit_score",
        "recommendation_rate_pct", "overall_sentiment_score",
        "mean_competency_score", "fit_trend",
        "concern_indicator_count", "concern_flag", "misalignment_flag",
    ] if col in filtered.columns]
    st.dataframe(
        filtered[table_cols].rename(columns={
            "agency_name_display": "Agency",
            "composite_fit_score": "Fit score",
            "concern_flag": "Flag",
            "misalignment_flag": "Misalignment",
            "overall_sentiment_score": "Sentiment",
            "recommendation_rate_pct": "Rec. rate (%)",
            "response_count": "Responses",
            "concern_indicator_count": "Signals",
            "mean_competency_score": "Mean comp. score",
            "fit_trend": "Trend",
        }),
        hide_index=True,
        use_container_width=True,
    )

    # trends section
    if trend_df is not None:
        st.markdown('<div class="section-header">Trends over time</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Agency-year trend data from agency_yearly_trends.csv.</div>',
            unsafe_allow_html=True,
        )

        # spotlight chart
        st.plotly_chart(chart_trend_spotlight(trend_df, profiles), use_container_width=True)

        # portfolio summary line chart
        portfolio_summary = summarize_portfolio_trends(trend_df, filtered)
        if not portfolio_summary.empty:
            trend_metric = st.radio(
                "Trend metric",
                ["Mean fit score", "Mean recommendation rate", "Flagged agencies", "Evaluation count"],
                horizontal=True,
            )
            st.plotly_chart(chart_portfolio_trend(portfolio_summary, trend_metric), use_container_width=True)

    # agency drill-down
    render_agency_detail(filtered, text_df, trend_df, profiles)


if __name__ == "__main__":
    main()
