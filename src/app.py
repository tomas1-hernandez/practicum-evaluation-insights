# app.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

app_title = "Practicum Evaluation Intelligence"
color_positive = "#4f97b7"
color_concern = "#e6616b"
color_neutral = "#7a7a7a"
color_bg = "#ffffff"

this_file = Path(__file__).resolve()
candidate_roots = [this_file.parent, this_file.parent.parent]
root_dir = next((p for p in candidate_roots if (p / "outputs").exists()), this_file.parent)

profile_candidates = [
    root_dir / "outputs" / "agency_profiles" / "agency_profiles.csv",
    root_dir / "agency_profiles.csv",
]
text_candidates = [
    root_dir / "outputs" / "tables" / "evaluations_text.csv",
    root_dir / "evaluations_text.csv",
]
trend_candidates = [
    root_dir / "outputs" / "tables" / "agency_yearly_trends.csv",
    root_dir / "agency_yearly_trends.csv",
]

theme_labels = {
    "strong_supervision_helpful_pct": "Strong supervision - helpful",
    "strong_supervision_least_pct": "Supervision concerns - least helpful",
    "direct_practice_opportunity_helpful_pct": "Direct practice - helpful",
    "direct_practice_opportunity_least_pct": "Direct practice lacking - least helpful",
    "administrative_overload_helpful_pct": "Admin work - helpful",
    "administrative_overload_least_pct": "Admin overload - least helpful",
    "organizational_structure_helpful_pct": "Structure - helpful",
    "organizational_structure_least_pct": "Structure issues - least helpful",
    "learning_environment_helpful_pct": "Learning environment - helpful",
    "learning_environment_least_pct": "Learning environment issues - least helpful",
    "social_justice_alignment_helpful_pct": "Social justice alignment - helpful",
    "social_justice_alignment_least_pct": "Social justice alignment gaps - least helpful",
}

detail_metrics = {
    "prepared_for_practice": "Prepared for practice",
    "learning_goals_met": "Learning goals met",
    "supervision_quality": "Supervision quality",
    "supervision_frequency": "Supervision frequency",
    "felt_prepared": "Felt prepared",
}

st.set_page_config(page_title=app_title, layout="wide")

st.markdown(
    """
    <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
        div[data-testid="metric-container"] {
            background-color: #F8F9FA;
            border: 1px solid #E5E7EB;
            border-radius: 8px;
            padding: 16px 18px;
        }
        .section-header {
            font-size: 1.1rem;
            font-weight: 700;
            margin-top: 1.2rem;
            margin-bottom: 0.25rem;
        }
        .section-subhead {
            color: #666666;
            margin-bottom: 0.75rem;
        }
        .note-box {
            background: #f8f9fa;
            border-left: 4px solid #4f97b7;
            padding: 0.8rem 1rem;
            border-radius: 4px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def first_existing_path(candidates):
    for path in candidates:
        if path.exists():
            return path
    return None


@st.cache_data
def load_profiles() -> pd.DataFrame:
    path = first_existing_path(profile_candidates)
    if path is None:
        st.error("Could not find agency_profiles.csv. Put it in outputs/agency_profiles/ or beside this app file.")
        st.stop()

    df = pd.read_csv(path)

    numeric_cols = (
        [
            "response_count",
            "composite_fit_score",
            "recommendation_rate",
            "overall_sentiment_score",
            "concern_indicator_count",
        ]
        + list(theme_labels.keys())
        + list(detail_metrics.keys())
    )

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "recommendation_rate" in df.columns and df["recommendation_rate"].dropna().max() <= 1.0:
        df["recommendation_rate_pct"] = df["recommendation_rate"] * 100
    else:
        df["recommendation_rate_pct"] = df["recommendation_rate"]
        df["recommendation_rate"] = df["recommendation_rate"] / 100

    df["concern_flag"] = df.get("concern_flag", "no flag").fillna("no flag")
    df["is_flagged"] = df["concern_flag"].str.lower().eq("review recommended")
    df["data_quality"] = df.get("data_quality", "unknown").fillna("unknown")
    df["agency_name_display"] = df.get("agency_name_display", df["agency_name"].astype(str).str.title())

    return df.sort_values(
        ["is_flagged", "composite_fit_score", "agency_name_display"], ascending=[False, True, True]
    ).reset_index(drop=True)


@st.cache_data
def load_text_data():
    path = first_existing_path(text_candidates)
    if path is None:
        return None
    return pd.read_csv(path)


@st.cache_data
def load_trend_data() -> pd.DataFrame | None:
    path = first_existing_path(trend_candidates)
    if path is None:
        return None

    df = pd.read_csv(path)
    numeric_cols = [
        "academic_year_start",
        "response_count",
        "composite_fit_score",
        "recommendation_rate",
        "overall_sentiment_score",
        "concern_indicator_count",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "recommendation_rate" in df.columns and df["recommendation_rate"].dropna().max() <= 1.0:
        df["recommendation_rate_pct"] = df["recommendation_rate"] * 100
    else:
        df["recommendation_rate_pct"] = df["recommendation_rate"]
        df["recommendation_rate"] = df["recommendation_rate"] / 100

    df["agency_name_display"] = df.get("agency_name_display", df["agency_name"].astype(str).str.title())
    df["concern_flag"] = df.get("concern_flag", "no flag").fillna("no flag")
    df["is_flagged"] = df["concern_flag"].str.lower().eq("review recommended")
    if "academic_year_start" not in df.columns:
        df["academic_year_start"] = df["academic_year"].astype(str).str.extract(r"(\d{4})").astype(float)
    return df.sort_values(["academic_year_start", "agency_name_display"])


def base_layout(fig: go.Figure, title: str, subtitle: str | None = None) -> go.Figure:
    title_text = f"<b>{title}</b>"
    if subtitle:
        title_text += f"<br><span style='font-size:12px;color:#666666'>{subtitle}</span>"
    fig.update_layout(
        template="plotly_white",
        title={"text": title_text, "x": 0, "xanchor": "left"},
        paper_bgcolor=color_bg,
        plot_bgcolor=color_bg,
        margin=dict(l=20, r=20, t=75, b=20),
        font=dict(size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def build_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    flag_view = st.sidebar.radio(
        "Flag status",
        ["All agencies", "Flagged only", "Not flagged only"],
    )

    min_responses = int(df["response_count"].min()) if "response_count" in df.columns else 1
    max_responses = int(df["response_count"].max()) if "response_count" in df.columns else 20

    response_floor = st.sidebar.slider(
        "Minimum responses",
        min_value=min_responses,
        max_value=max_responses,
        value=max(3, min_responses),
    )

    fit_min = float(df["composite_fit_score"].min())
    fit_max = float(df["composite_fit_score"].max())

    fit_range = st.sidebar.slider(
        "Composite fit score",
        min_value=round(fit_min, 1),
        max_value=round(fit_max, 1),
        value=(round(fit_min, 1), round(fit_max, 1)),
        step=0.1,
    )

    search = st.sidebar.text_input("Search agency", "")

    filtered = df.copy()
    filtered = filtered[filtered["response_count"] >= response_floor]
    filtered = filtered[
        (filtered["composite_fit_score"] >= fit_range[0]) & (filtered["composite_fit_score"] <= fit_range[1])
    ]

    if flag_view == "Flagged only":
        filtered = filtered[filtered["is_flagged"]]
    elif flag_view == "Not flagged only":
        filtered = filtered[~filtered["is_flagged"]]

    if search.strip():
        filtered = filtered[filtered["agency_name_display"].str.contains(search.strip(), case=False, na=False)]

    st.sidebar.markdown("---")
    st.sidebar.caption("This app reads the built agency profile table. It does not rerun the pipeline.")
    return filtered.reset_index(drop=True)


def render_kpis(df: pd.DataFrame) -> None:
    total = len(df)
    flagged = int(df["is_flagged"].sum())
    pct_flagged = round((flagged / total) * 100, 1) if total else 0
    mean_fit = round(df["composite_fit_score"].mean(), 2) if total else 0
    mean_rec = round(df["recommendation_rate_pct"].mean(), 1) if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Agencies in current view", total)
    c2.metric("Review recommended", flagged, f"{pct_flagged}% of current view")
    c3.metric("Mean fit score", mean_fit)
    c4.metric("Mean recommendation rate", f"{mean_rec}%")

    st.markdown(
        """
        <div class="note-box">
        This dashboard is meant to support leadership review, not make automatic decisions.
        Agencies are best interpreted alongside response count, qualitative themes, and program context.
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_flag_summary(df: pd.DataFrame) -> go.Figure:
    summary = pd.DataFrame(
        {
            "Flag status": ["No flag", "Review recommended"],
            "Count": [int((~df["is_flagged"]).sum()), int(df["is_flagged"].sum())],
        }
    )
    fig = px.bar(
        summary,
        x="Count",
        y="Flag status",
        orientation="h",
        text="Count",
        color="Flag status",
        color_discrete_map={"No flag": color_positive, "Review recommended": color_concern},
    )
    fig = base_layout(fig, "How many agencies are currently flagged for review")
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False)
    return fig


def chart_fit_vs_recommend(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df,
        x="composite_fit_score",
        y="recommendation_rate_pct",
        color="concern_flag",
        hover_name="agency_name_display",
        hover_data={"response_count": True, "overall_sentiment_score": ":.2f"},
        color_discrete_map={"no flag": color_positive, "review recommended": color_concern},
        symbol="concern_flag",
        symbol_map={"no flag": "circle", "review recommended": "triangle-up"},
    )
    fig.add_vline(x=3.5, line_dash="dot", line_color=color_neutral)
    fig.add_hline(y=70, line_dash="dash", line_color=color_neutral)
    fig = base_layout(
        fig,
        "Agencies with lower fit scores also tend to receive fewer recommendations",
        "Reference lines show the current fit and recommendation cut points used for concern flagging",
    )
    fig.update_xaxes(title="Composite fit score", range=[2.0, 5.1])
    fig.update_yaxes(title="Recommendation rate (%)", range=[0, 105])
    return fig


def chart_theme_heatmap(df: pd.DataFrame) -> go.Figure:
    available = [c for c in theme_labels if c in df.columns]
    long_df = pd.DataFrame(
        {
            "Theme": [theme_labels[c] for c in available],
            "Average share of agency responses": [df[c].mean() for c in available],
        }
    ).sort_values("Average share of agency responses", ascending=True)

    fig = px.bar(
        long_df,
        x="Average share of agency responses",
        y="Theme",
        orientation="h",
        text="Average share of agency responses",
    )
    fig = base_layout(
        fig,
        "Which themes appear most often across agency profiles",
        "Helpful and least-helpful theme rates are averaged separately, then compared across agencies",
    )
    fig.update_traces(marker_color=color_positive, texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_xaxes(title="Average share of responses tagged (%)")
    fig.update_layout(showlegend=False)
    return fig


def chart_bottom_ten(df: pd.DataFrame) -> go.Figure:
    bottom = df.sort_values("composite_fit_score", ascending=True).head(10).copy()
    bottom["label"] = bottom["agency_name_display"] + " (n=" + bottom["response_count"].astype(str) + ")"
    fig = px.bar(bottom, x="composite_fit_score", y="label", orientation="h", text="composite_fit_score")
    fig = base_layout(
        fig,
        "Lowest composite fit scores in the current view",
        "Response count is shown beside each agency name to reduce over-reading small-N results",
    )
    fig.update_traces(marker_color=color_concern, texttemplate="%{text:.2f}", textposition="outside")
    fig.update_xaxes(title="Composite fit score", range=[0, 5])
    fig.update_yaxes(title="")
    fig.update_layout(showlegend=False)
    return fig


def summarize_top_phrases(value: str, limit: int = 8) -> str:
    if not isinstance(value, str) or not value.strip():
        return "No phrase summary available."
    parts = [part.strip() for part in value.split("|") if part.strip()]
    cleaned = []
    for part in parts[:limit]:
        phrase = part.split(":")[0].strip()
        cleaned.append(phrase)
    return ", ".join(cleaned) if cleaned else "No phrase summary available."


def summarize_portfolio_trends(trend_df: pd.DataFrame, filtered_profiles: pd.DataFrame) -> pd.DataFrame:
    filtered_names = filtered_profiles["agency_name"].unique().tolist()
    subset = trend_df[trend_df["agency_name"].isin(filtered_names)].copy()
    if subset.empty:
        return subset

    rows = []
    for academic_year, group in subset.groupby("academic_year"):
        response_total = group["response_count"].sum()
        rows.append(
            {
                "academic_year": academic_year,
                "academic_year_start": group["academic_year_start"].iloc[0],
                "evaluations": int(response_total),
                "agencies": int(group["agency_name"].nunique()),
                "flagged_agencies": int(group["is_flagged"].sum()),
                "mean_fit_score": round(
                    (group["composite_fit_score"] * group["response_count"]).sum() / response_total, 2
                ),
                "mean_recommendation_rate_pct": round(
                    (group["recommendation_rate_pct"] * group["response_count"]).sum() / response_total, 1
                ),
                "mean_sentiment_score": round(
                    (group["overall_sentiment_score"] * group["response_count"]).sum() / response_total, 3
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("academic_year_start")


def chart_portfolio_trend(summary_df: pd.DataFrame, metric: str) -> go.Figure:
    metric_map = {
        "Mean fit score": ("mean_fit_score", "Mean fit score", [2.5, 5.0]),
        "Mean recommendation rate": ("mean_recommendation_rate_pct", "Recommendation rate (%)", [0, 100]),
        "Flagged agencies": ("flagged_agencies", "Flagged agencies", None),
        "Evaluation count": ("evaluations", "Evaluations", None),
    }
    metric_col, axis_label, axis_range = metric_map[metric]
    fig = px.line(summary_df, x="academic_year", y=metric_col, markers=True, text=metric_col)
    fig = base_layout(
        fig,
        f"Portfolio trend: {metric.lower()}",
        "This trend is built from the dedicated agency_yearly_trends.csv output",
    )
    fig.update_traces(line_color=color_positive, textposition="top center")
    if metric == "Mean fit score":
        fig.add_hline(y=3.5, line_dash="dot", line_color=color_neutral)
        fig.update_traces(texttemplate="%{text:.2f}")
    elif metric == "Mean recommendation rate":
        fig.add_hline(y=70, line_dash="dash", line_color=color_neutral)
        fig.update_traces(texttemplate="%{text:.1f}%")
    elif metric == "Flagged agencies":
        fig.update_traces(texttemplate="%{text:.0f}")
    else:
        fig.update_traces(texttemplate="%{text:.0f}")
    fig.update_yaxes(title=axis_label, range=axis_range)
    fig.update_xaxes(title="Academic year")
    return fig


def chart_agency_trend(agency_trend_df: pd.DataFrame, agency_label: str) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=agency_trend_df["academic_year"],
            y=agency_trend_df["composite_fit_score"],
            mode="lines+markers+text",
            text=[f"{value:.2f}" for value in agency_trend_df["composite_fit_score"]],
            textposition="top center",
            name="Fit score",
            line=dict(color=color_positive, width=3),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=agency_trend_df["academic_year"],
            y=agency_trend_df["recommendation_rate_pct"],
            name="Recommendation rate (%)",
            marker_color="#dbe8ee",
            opacity=0.8,
        ),
        secondary_y=True,
    )
    fig = base_layout(
        fig,
        f"Over-time view for {agency_label}",
        "The bars represent the recommendation rate, and the line shows the fit score. Each point is one agency-year entry from the yearly trends file.",
    )
    fig.add_hline(y=3.5, line_dash="dot", line_color=color_neutral, secondary_y=False)
    fig.add_hline(y=70, line_dash="dash", line_color=color_neutral, secondary_y=True)
    fig.update_yaxes(title_text="Composite fit score", range=[1, 5], secondary_y=False)
    fig.update_yaxes(title_text="Recommendation rate (%)", range=[0, 100], secondary_y=True)
    fig.update_xaxes(title="Academic year")
    return fig


def render_agency_detail(filtered: pd.DataFrame, text_df: pd.DataFrame | None, trend_df: pd.DataFrame | None) -> None:
    st.markdown('<div class="section-header">Agency drill-down</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subhead">Pick one agency to review its score profile, response count, top phrases, and over-time pattern.</div>',
        unsafe_allow_html=True,
    )

    agency_label = st.selectbox("Select an agency", filtered["agency_name_display"].tolist())
    row = filtered.loc[filtered["agency_name_display"] == agency_label].iloc[0]

    left, right = st.columns([1.1, 1.2])

    with left:
        st.subheader(agency_label)
        st.write(f"**Flag status:** {row['concern_flag'].title()}")
        st.write(f"**Responses:** {int(row['response_count'])}")
        st.write(f"**Data quality:** {row['data_quality']}")
        st.write(f"**Composite fit score:** {row['composite_fit_score']:.2f}")
        st.write(f"**Recommendation rate:** {row['recommendation_rate_pct']:.1f}%")
        st.write(f"**Overall sentiment score:** {row['overall_sentiment_score']:.2f}")
        st.write(f"**Concern indicators triggered:** {int(row['concern_indicator_count'])}")

    with right:
        metric_columns = st.columns(3)
        key_metrics = [col for col in detail_metrics if col in row.index]
        for idx, col in enumerate(key_metrics[:3]):
            metric_columns[idx].metric(detail_metrics[col], f"{row[col]:.2f}")
        if len(key_metrics) > 3:
            extra_columns = st.columns(max(1, len(key_metrics) - 3))
            for idx, col in enumerate(key_metrics[3:]):
                extra_columns[idx].metric(detail_metrics[col], f"{row[col]:.2f}")

    theme_cols = [c for c in theme_labels if c in filtered.columns]
    if theme_cols:
        st.markdown("**Theme profile**")
        theme_table = pd.DataFrame(
            {
                "Theme": [theme_labels[c] for c in theme_cols],
                "Percent": [row[c] for c in theme_cols],
            }
        ).sort_values("Percent", ascending=False)
        st.dataframe(theme_table, use_container_width=True, hide_index=True)

    p1, p2 = st.columns(2)
    with p1:
        st.markdown("**Top helpful phrases**")
        st.write(summarize_top_phrases(row.get("most_helpful_top_bigrams")))
    with p2:
        st.markdown("**Top least-helpful phrases**")
        st.write(summarize_top_phrases(row.get("least_helpful_top_bigrams")))

    if trend_df is not None:
        agency_trend = trend_df[trend_df["agency_name"] == row["agency_name"]].copy()
        if len(agency_trend) > 0:
            st.plotly_chart(
                chart_agency_trend(agency_trend.sort_values("academic_year_start"), agency_label),
                use_container_width=True,
            )
            trend_table = agency_trend[
                [
                    "academic_year",
                    "response_count",
                    "composite_fit_score",
                    "recommendation_rate_pct",
                    "overall_sentiment_score",
                    "concern_flag",
                ]
            ].rename(
                columns={
                    "academic_year": "Academic year",
                    "response_count": "Responses",
                    "composite_fit_score": "Fit score",
                    "recommendation_rate_pct": "Recommendation rate (%)",
                    "overall_sentiment_score": "Sentiment",
                    "concern_flag": "Flag",
                }
            )
            st.dataframe(trend_table, use_container_width=True, hide_index=True)

    if text_df is not None and "agency_name" in text_df.columns:
        matches = text_df[text_df["agency_name"] == row["agency_name"]]
        if not matches.empty:
            st.markdown("**Example evaluation text**")
            preview_cols = [
                col
                for col in ["academic_year", "most_helpful", "least_helpful", "recommend_reason", "comments_supervisor"]
                if col in matches.columns
            ]
            preview = matches[preview_cols].head(5).copy()
            st.dataframe(preview, use_container_width=True, hide_index=True)


def main() -> None:
    profiles = load_profiles()
    text_df = load_text_data()
    trend_df = load_trend_data()

    st.title(app_title)
    st.caption("Leadership-facing review dashboard built from agency profile outputs")

    filtered = build_sidebar(profiles)
    if filtered.empty:
        st.warning("No agencies match the current filter settings. Adjust the sidebar filters and try again.")
        return

    render_kpis(filtered)

    st.markdown('<div class="section-header">Portfolio view</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subhead">These charts summarize the agencies that remain after the current sidebar filters are applied.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    c1.plotly_chart(chart_flag_summary(filtered), use_container_width=True)
    c2.plotly_chart(chart_fit_vs_recommend(filtered), use_container_width=True)

    c3, c4 = st.columns(2)
    c3.plotly_chart(chart_theme_heatmap(filtered), use_container_width=True)
    c4.plotly_chart(chart_bottom_ten(filtered), use_container_width=True)

    table_cols = [
        "agency_name_display",
        "response_count",
        "composite_fit_score",
        "recommendation_rate_pct",
        "overall_sentiment_score",
        "concern_indicator_count",
        "concern_flag",
    ]
    st.markdown("**Agency summary table**")
    st.dataframe(
        filtered[table_cols].rename(
            columns={
                "agency_name_display": "Agency",
                "response_count": "Responses",
                "composite_fit_score": "Fit score",
                "recommendation_rate_pct": "Recommendation rate (%)",
                "overall_sentiment_score": "Sentiment",
                "concern_indicator_count": "Signals",
                "concern_flag": "Flag",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    if trend_df is not None:
        st.markdown('<div class="section-header">Trends over time</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subhead">This section reads the dedicated agency_yearly_trends.csv output. It does not recompute trends from the raw file inside the app.</div>',
            unsafe_allow_html=True,
        )
        portfolio_summary = summarize_portfolio_trends(trend_df, filtered)
        if not portfolio_summary.empty:
            trend_metric = st.radio(
                "Trend metric",
                ["Mean fit score", "Mean recommendation rate", "Flagged agencies", "Evaluation count"],
                horizontal=True,
            )
            st.plotly_chart(chart_portfolio_trend(portfolio_summary, trend_metric), use_container_width=True)
            st.dataframe(
                portfolio_summary.rename(
                    columns={
                        "academic_year": "Academic year",
                        "evaluations": "Evaluations",
                        "agencies": "Agencies",
                        "flagged_agencies": "Flagged agencies",
                        "mean_fit_score": "Mean fit score",
                        "mean_recommendation_rate_pct": "Mean recommendation rate (%)",
                        "mean_sentiment_score": "Mean sentiment score",
                    }
                ).drop(columns=["academic_year_start"]),
                use_container_width=True,
                hide_index=True,
            )

    render_agency_detail(filtered, text_df, trend_df)


if __name__ == "__main__":
    main()
