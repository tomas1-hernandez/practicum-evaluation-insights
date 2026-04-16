"""Build the static report figures for practicum evaluation intelligence.

This script reads the pipeline outputs and saves the figures used in the
written report, deck, and portfolio piece. It runs separately from the
main pipeline so visuals can be refreshed without rebuilding every table.

Run this directly after the pipeline finishes:
    python create_visualizations.py

All figures are saved to outputs/figures/ as PNG files.
"""

from __future__ import annotations

import random
import re
from collections import Counter

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from wordcloud import WordCloud

from config import figure_dpi, figures_dir, input_file, profiles_dir, tables_dir
from constants import fit_score_cols, likert_cols, likert_display, stopwords

# ------------------------------------------------------------------------------
# Color palette - one source of truth for the whole figure set
# ------------------------------------------------------------------------------

COLORS = {
    "concern": "#c0392b",      # strong red for flagged / problematic
    "caution": "#e67e22",      # orange for borderline
    "neutral": "#7f8c8d",      # gray for gridlines and reference lines
    "positive": "#2471a3",     # steel blue for good / no flag
    "light_blue": "#aad2e4",   # lighter blue for volume bars
    "background": "#ffffff",
}

# ------------------------------------------------------------------------------
# Global matplotlib style - applied once, inherited by all figures
# ------------------------------------------------------------------------------

plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "figure.dpi": 100,
        "figure.facecolor": COLORS["background"],
        "font.family": "sans-serif",
        "font.size": 10,
        "grid.alpha": 0.2,
        "grid.color": COLORS["neutral"],
    }
)

# file paths - read from config so nothing is hardcoded here
profiles_file = profiles_dir / "agency_profiles.csv"
trends_file = tables_dir / "agency_yearly_trends.csv"


# ------------------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------------------


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Save a figure with consistent export settings and close it after."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        figures_dir / filename,
        bbox_inches="tight",
        dpi=figure_dpi,
        edgecolor="none",
        facecolor=COLORS["background"],
    )
    plt.close(fig)


def build_word_freq_dict(text_series: pd.Series, top_n: int = 150) -> dict[str, int]:
    """Build a word frequency dictionary for word clouds from a text column."""
    tokens: list[str] = []
    for text in text_series.dropna():
        words = re.findall(r"\b[a-z]+\b", str(text).lower())
        tokens.extend(
            [word for word in words if word not in stopwords and len(word) > 2]
        )
    return dict(Counter(tokens).most_common(top_n))


def parse_bigram_string(value: str, top_n: int = 10) -> dict[str, int]:
    """Parse a saved bigram summary string back into a word-count dictionary."""
    if not isinstance(value, str):
        return {}
    result: dict[str, int] = {}
    for item in value.split(" | ")[:top_n]:
        parts = item.rsplit(":", 1)
        if len(parts) == 2:
            try:
                result[parts[0].strip()] = int(parts[1].strip())
            except ValueError:
                continue
    return result


def get_display_col(df: pd.DataFrame) -> str:
    """Use the cleaned display name when available, fall back to raw name."""
    return "agency_name_display" if "agency_name_display" in df.columns else "agency_name"


def blue_color_func(*args, **kwargs) -> str:
    """Return a blue shade for positive word clouds."""
    return f"hsl(210, 80%, {random.randint(20, 50)}%)"


def red_color_func(*args, **kwargs) -> str:
    """Return a red shade for concern word clouds."""
    return f"hsl(5, 80%, {random.randint(20, 50)}%)"


# ------------------------------------------------------------------------------
# Individual figure functions
# One function per figure keeps the code easy to read and easy to rerun
# individually when you only need to refresh one chart.
# ------------------------------------------------------------------------------


def fig_01_concern_flag_summary(sufficient_df: pd.DataFrame) -> None:
    """Horizontal bar chart: how many agencies are flagged vs not flagged."""
    flagged = int((sufficient_df["concern_flag"] == "review recommended").sum())
    not_flagged = len(sufficient_df) - flagged
    flagged_share = round((flagged / len(sufficient_df)) * 100, 1) if len(sufficient_df) else 0

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(
        ["No flag", "Review recommended"],
        [not_flagged, flagged],
        alpha=0.88,
        color=[COLORS["positive"], COLORS["concern"]],
    )
    for bar, count in zip(bars, [not_flagged, flagged]):
        ax.text(
            count + 1,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="center",
        )
    ax.set_xlabel("Number of agencies")
    ax.set_xlim(0, max(not_flagged, flagged) + 25)
    ax.set_title(
        f"Figure 1. {flagged_share}% of practicum agencies meet the threshold for a leadership review",
        loc="left",
    )
    plt.tight_layout()
    save_figure(fig, "fig_01_concern_flag_summary.png")


def fig_02_fit_vs_recommendation(sufficient_df: pd.DataFrame) -> None:
    """Scatter plot: placement quality score vs recommendation rate by flag status."""
    mask = sufficient_df["concern_flag"] == "review recommended"
    recommendation_pct = sufficient_df["recommendation_rate"] * 100

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        sufficient_df.loc[~mask, "composite_fit_score"],
        recommendation_pct.loc[~mask],
        alpha=0.6,
        color=COLORS["positive"],
        label="No flag",
        s=50,
        zorder=3,
    )
    ax.scatter(
        sufficient_df.loc[mask, "composite_fit_score"],
        recommendation_pct.loc[mask],
        alpha=0.75,
        color=COLORS["concern"],
        label="Flagged for review",
        marker="^",
        s=65,
        zorder=4,
    )
    ax.axhline(70, alpha=0.5, color=COLORS["neutral"], linestyle="--", linewidth=1)
    ax.axvline(3.5, alpha=0.5, color=COLORS["neutral"], linestyle=":", linewidth=1)
    ax.text(4.7, 68, "70% threshold", color=COLORS["neutral"], fontsize=8, va="top")
    ax.text(3.52, 15, "3.5 cut point", color=COLORS["neutral"], fontsize=8, rotation=90, va="bottom")
    ax.set_xlabel("Placement Quality Score (1-5)")
    ax.set_ylabel("Recommendation rate (%)")
    ax.set_title(
        "Figure 2. Agencies with lower placement quality scores tend to get fewer\n"
        "recommendations — most flagged agencies cluster in the lower-left quadrant",
        loc="left",
    )
    ax.legend(fontsize=9, frameon=False)
    plt.tight_layout()
    save_figure(fig, "fig_02_fit_vs_recommendation.png")


def fig_03_sentiment_distribution(sufficient_df: pd.DataFrame) -> None:
    """Overlapping histograms: sentiment score by concern flag status."""
    flagged = sufficient_df[sufficient_df["concern_flag"] == "review recommended"]["overall_sentiment_score"]
    not_flagged = sufficient_df[sufficient_df["concern_flag"] == "no flag"]["overall_sentiment_score"]
    overall_mean = sufficient_df["overall_sentiment_score"].mean()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(not_flagged, alpha=0.75, bins=25, color=COLORS["positive"], label=f"No flag ({len(not_flagged)} agencies)")
    ax.hist(flagged, alpha=0.75, bins=25, color=COLORS["concern"], label=f"Flagged ({len(flagged)} agencies)")
    ax.axvline(overall_mean, color=COLORS["neutral"], label=f"Overall mean ({overall_mean:.3f})", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Overall sentiment score")
    ax.set_ylabel("Number of agencies")
    ax.set_title(
        "Figure 3. Flagged agencies skew toward lower sentiment scores, though the two groups still overlap",
        loc="left",
    )
    ax.legend(frameon=False)
    plt.tight_layout()
    save_figure(fig, "fig_03_sentiment_distribution.png")


def fig_04_likert_mean_scores(sufficient_df: pd.DataFrame) -> None:
    """Horizontal bar chart: mean Likert scores sorted ascending so lowest is at bottom."""
    mean_scores = sufficient_df[likert_cols].mean().round(2)
    mean_scores = mean_scores.sort_values(ascending=True)

    colors = [
        COLORS["concern"] if score < 3.7
        else COLORS["caution"] if score < 3.9
        else COLORS["positive"]
        for score in mean_scores
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        [likert_display.get(col, col) for col in mean_scores.index],
        mean_scores,
        alpha=0.88,
        color=colors,
    )
    ax.axvline(mean_scores.mean(), color="#121212", linestyle=":", linewidth=1.5, alpha=0.6)
    ax.set_xlim(1, 5.4)
    ax.set_xlabel("Mean score (1 = strongly disagree, 5 = strongly agree)")
    ax.set_title(
        "Figure 4. Students rate supervision highly, but preparedness and direct practice items score lower",
        loc="left",
    )
    for bar, score in zip(bars, mean_scores):
        ax.text(
            score + 0.06,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.2f}",
            fontsize=8.5,
            ha="left",
            va="center",
        )
    ax.legend(
        handles=[
            mpatches.Patch(color=COLORS["concern"], label="Below 3.70"),
            mpatches.Patch(color=COLORS["caution"], label="3.70 to 3.89"),
            mpatches.Patch(color=COLORS["positive"], label="3.90 and above"),
        ],
        fontsize=8,
        frameon=False,
        loc="lower right",
    )
    plt.tight_layout()
    save_figure(fig, "fig_04_likert_mean_scores.png")


def fig_05_theme_frequency(sufficient_df: pd.DataFrame) -> None:
    """Grouped bar chart: theme rates sorted by gap so most problematic themes appear first."""
    theme_names = sorted(
        {col.replace("_helpful_pct", "") for col in sufficient_df.columns if col.endswith("_helpful_pct")}
    )
    helpful = {name: sufficient_df[f"{name}_helpful_pct"].mean() for name in theme_names}
    least = {name: sufficient_df[f"{name}_least_pct"].mean() for name in theme_names}
    gaps = {name: least[name] - helpful[name] for name in theme_names}
    sorted_names = sorted(theme_names, key=lambda n: gaps[n], reverse=True)
    labels = [name.replace("_", " ") for name in sorted_names]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = range(len(labels))
    width = 0.38
    ax.bar(
        [v - width / 2 for v in x],
        [helpful[n] for n in sorted_names],
        width=width,
        alpha=0.88,
        color=COLORS["positive"],
        label="Appeared in most helpful",
    )
    ax.bar(
        [v + width / 2 for v in x],
        [least[n] for n in sorted_names],
        width=width,
        alpha=0.88,
        color=COLORS["concern"],
        label="Appeared in least helpful",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Mean % of responses tagged")
    ax.set_title(
        "Figure 5. Administrative overload appears far more often in least-helpful comments\n"
        "while strong supervision and direct practice drive most-helpful responses",
        loc="left",
    )
    ax.legend(frameon=False)
    ax.yaxis.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(fig, "fig_05_theme_frequency.png")


def fig_06_bigrams_comparison(profiles_df: pd.DataFrame) -> None:
    """Side-by-side horizontal bar charts: top bigrams in helpful vs least-helpful."""
    helpful_bigrams: Counter = Counter()
    least_bigrams: Counter = Counter()
    for value in profiles_df["most_helpful_top_bigrams"].dropna():
        helpful_bigrams.update(parse_bigram_string(value, 15))
    for value in profiles_df["least_helpful_top_bigrams"].dropna():
        least_bigrams.update(parse_bigram_string(value, 15))

    top_helpful = helpful_bigrams.most_common(12)
    top_least = least_bigrams.most_common(12)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))
    ax_left.barh(
        [item[0] for item in top_helpful],
        [item[1] for item in top_helpful],
        alpha=0.88,
        color=COLORS["positive"],
    )
    ax_left.invert_yaxis()
    ax_left.set_title("Most helpful - top phrases", loc="left", fontweight="bold")
    ax_left.tick_params(axis="y", labelsize=9)
    ax_left.xaxis.grid(True, alpha=0.2)
    ax_left.set_axisbelow(True)

    ax_right.barh(
        [item[0] for item in top_least],
        [item[1] for item in top_least],
        alpha=0.88,
        color=COLORS["concern"],
    )
    ax_right.invert_yaxis()
    ax_right.set_title("Least helpful - top phrases", loc="left", fontweight="bold")
    ax_right.tick_params(axis="y", labelsize=9)
    ax_right.xaxis.grid(True, alpha=0.2)
    ax_right.set_axisbelow(True)

    fig.suptitle(
        "Figure 6. Helpful comments center on confidence and real practice;\n"
        "least-helpful comments center on admin work and limited client contact",
        fontsize=11,
        fontweight="bold",
        ha="left",
        x=0.02,
        y=1.02,
    )
    plt.tight_layout()
    save_figure(fig, "fig_06_bigrams_comparison.png")


def fig_07_wordcloud_most_helpful(row_df: pd.DataFrame) -> None:
    """Word cloud: most common words in most-helpful responses."""
    helpful_freq = build_word_freq_dict(row_df["most_helpful"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(
        WordCloud(
            background_color="white",
            color_func=blue_color_func,
            height=500,
            max_words=80,
            width=1000,
        ).generate_from_frequencies(helpful_freq),
        interpolation="bilinear",
    )
    ax.axis("off")
    ax.set_title(
        "Figure 7. Students most often describe support, learning, and hands-on practice\n"
        "as the most helpful parts of their placement",
        loc="left",
        pad=12,
    )
    save_figure(fig, "fig_07_wordcloud_most_helpful.png")


def fig_08_wordcloud_least_helpful(row_df: pd.DataFrame) -> None:
    """Word cloud: most common words in least-helpful responses."""
    least_freq = build_word_freq_dict(row_df["least_helpful"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(
        WordCloud(
            background_color="white",
            color_func=red_color_func,
            height=500,
            max_words=80,
            width=1000,
        ).generate_from_frequencies(least_freq),
        interpolation="bilinear",
    )
    ax.axis("off")
    ax.set_title(
        "Figure 8. Student frustrations most often mention administrative work,\n"
        "limited direct practice, and not enough client contact",
        loc="left",
        pad=12,
    )
    save_figure(fig, "fig_08_wordcloud_least_helpful.png")


def fig_09_top_flagged_agencies(sufficient_df: pd.DataFrame) -> None:
    """Horizontal bar chart: the 10 flagged agencies with the lowest fit scores."""
    display_col = get_display_col(sufficient_df)
    flagged_df = (
        sufficient_df[sufficient_df["concern_flag"] == "review recommended"]
        .sort_values("composite_fit_score", ascending=True)
        .head(10)
        .copy()
    )
    flagged_df["label"] = (
        flagged_df[display_col] + "  (n=" + flagged_df["response_count"].astype(str) + ")"
    )

    fig, ax = plt.subplots(figsize=(10, 6.5))
    bars = ax.barh(flagged_df["label"], flagged_df["composite_fit_score"], alpha=0.88, color=COLORS["concern"])
    ax.axvline(3.5, alpha=0.6, color=COLORS["neutral"], linestyle="--", linewidth=1, label="Low-fit threshold (3.5)")
    ax.set_xlim(0, 5)
    ax.set_xlabel("Placement Quality Score (1-5)")
    ax.set_title(
        "Figure 9. The ten flagged agencies with the lowest Placement Quality Scores",
        loc="left",
    )
    ax.tick_params(axis="y", labelsize=8.5)
    ax.invert_yaxis()
    for bar, score in zip(bars, flagged_df["composite_fit_score"]):
        ax.text(
            score + 0.06,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.2f}",
            fontsize=8.5,
            ha="left",
            va="center",
        )
    ax.legend(frameon=False)
    ax.xaxis.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(fig, "fig_09_top_flagged_agencies.png")


def fig_10_yearly_fit_trend(trends_df: pd.DataFrame) -> None:
    """Line chart: mean placement quality score by academic year.

    Separated from the volume bar chart (fig_11) so each story is clean
    and both render well on Streamlit without stacking.
    """
    yearly = (
        trends_df.groupby("academic_year")
        .apply(
            lambda g: pd.Series(
                {
                    "academic_year_start": g["academic_year_start"].iloc[0],
                    "mean_fit_score": round(
                        (g["composite_fit_score"] * g["response_count"]).sum()
                        / g["response_count"].sum(),
                        2,
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
        .sort_values("academic_year_start")
    )

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(
        yearly["academic_year"],
        yearly["mean_fit_score"],
        color=COLORS["positive"],
        linewidth=2.5,
        marker="o",
        markersize=6,
        zorder=3,
    )
    ax.axhline(3.5, alpha=0.6, color=COLORS["neutral"], linestyle=":", linewidth=1)
    ax.text(
        yearly["academic_year"].iloc[-1],
        3.52,
        "Low-fit threshold (3.5)",
        color=COLORS["neutral"],
        fontsize=8,
        ha="right",
        va="bottom",
    )
    ax.set_ylim(3.0, 4.8)
    ax.set_ylabel("Mean Placement Quality Score")
    ax.set_xlabel("Academic year")
    ax.tick_params(axis="x", rotation=40)
    ax.yaxis.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    for year, score in zip(yearly["academic_year"], yearly["mean_fit_score"]):
        ax.text(year, score + 0.06, f"{score:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(
        "Figure 10. Mean Placement Quality Score by academic year\n"
        "(weighted by number of responses per agency per year)",
        loc="left",
    )
    plt.tight_layout()
    save_figure(fig, "fig_10_yearly_fit_trend.png")


def fig_11_yearly_evaluation_volume(trends_df: pd.DataFrame) -> None:
    """Bar chart: number of evaluations submitted per academic year.

    Separated from the fit score line chart (fig_10) so each tells its
    own clear story and performs well on Streamlit.
    """
    yearly = (
        trends_df.groupby("academic_year")
        .apply(
            lambda g: pd.Series(
                {
                    "academic_year_start": g["academic_year_start"].iloc[0],
                    "evaluations": int(g["response_count"].sum()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
        .sort_values("academic_year_start")
    )

    fig, ax = plt.subplots(figsize=(11, 4.5))
    bars = ax.bar(
        yearly["academic_year"],
        yearly["evaluations"],
        alpha=0.9,
        color=COLORS["light_blue"],
    )
    ax.set_ylabel("Number of evaluations submitted")
    ax.set_xlabel("Academic year")
    ax.tick_params(axis="x", rotation=40)
    ax.yaxis.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    for bar, count in zip(bars, yearly["evaluations"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            str(count),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_title(
        "Figure 11. Evaluation volume submitted per academic year",
        loc="left",
    )
    plt.tight_layout()
    save_figure(fig, "fig_11_yearly_evaluation_volume.png")


def fig_12_fit_score_distribution(sufficient_df: pd.DataFrame) -> None:
    """Histogram: distribution of Placement Quality Scores — EDA figure.

    Gives the reader a feel for the full spread of scores before any
    grouping or filtering is applied.
    """
    scores = sufficient_df["composite_fit_score"].dropna()
    mean_score = scores.mean()
    median_score = scores.median()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(scores, bins=20, alpha=0.88, color=COLORS["positive"], edgecolor="white")
    ax.axvline(mean_score, color=COLORS["concern"], linestyle="--", linewidth=1.5, label=f"Mean: {mean_score:.2f}")
    ax.axvline(median_score, color=COLORS["neutral"], linestyle=":", linewidth=1.5, label=f"Median: {median_score:.2f}")
    ax.axvline(3.5, color=COLORS["caution"], linestyle="-", linewidth=1, alpha=0.7, label="Low-fit threshold: 3.5")
    ax.set_xlabel("Placement Quality Score (1-5)")
    ax.set_ylabel("Number of agencies")
    ax.set_title(
        "Figure 12. Distribution of Placement Quality Scores across all agencies with sufficient data",
        loc="left",
    )
    ax.legend(frameon=False)
    ax.yaxis.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(fig, "fig_12_fit_score_distribution.png")


def fig_13_recommendation_rate_distribution(sufficient_df: pd.DataFrame) -> None:
    """Histogram: distribution of recommendation rates — EDA figure."""
    rates = (sufficient_df["recommendation_rate"] * 100).dropna()
    mean_rate = rates.mean()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(rates, bins=20, alpha=0.88, color=COLORS["positive"], edgecolor="white")
    ax.axvline(mean_rate, color=COLORS["concern"], linestyle="--", linewidth=1.5, label=f"Mean: {mean_rate:.1f}%")
    ax.axvline(70, color=COLORS["caution"], linestyle="-", linewidth=1, alpha=0.7, label="Review threshold: 70%")
    ax.set_xlabel("Recommendation rate (%)")
    ax.set_ylabel("Number of agencies")
    ax.set_title(
        "Figure 13. Distribution of recommendation rates across all agencies with sufficient data",
        loc="left",
    )
    ax.legend(frameon=False)
    ax.yaxis.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(fig, "fig_13_recommendation_rate_distribution.png")


def fig_14_sentiment_score_distribution(sufficient_df: pd.DataFrame) -> None:
    """Histogram: distribution of overall sentiment scores — EDA figure."""
    scores = sufficient_df["overall_sentiment_score"].dropna()
    mean_score = scores.mean()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(scores, bins=20, alpha=0.88, color=COLORS["positive"], edgecolor="white")
    ax.axvline(mean_score, color=COLORS["concern"], linestyle="--", linewidth=1.5, label=f"Mean: {mean_score:.3f}")
    ax.axvline(0.05, color=COLORS["caution"], linestyle="-", linewidth=1, alpha=0.7, label="Concern threshold: 0.05")
    ax.set_xlabel("Overall sentiment score (VADER compound)")
    ax.set_ylabel("Number of agencies")
    ax.set_title(
        "Figure 14. Distribution of sentiment scores — most agencies cluster in mildly positive territory",
        loc="left",
    )
    ax.legend(frameon=False)
    ax.yaxis.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(fig, "fig_14_sentiment_distribution_eda.png")


def fig_15_agency_trend_spotlight(
    trends_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
) -> None:
    """Gray-out trend chart: all agency lines in gray, top flagged agencies in red.

    Inspired by the Economist / Cedric Scherer technique of graying out
    everything that is not the story and bringing the story forward in color.
    Every reader should immediately see which agencies are declining and roughly
    when the slide started — without reading a single number first.

    Layout:
        - All agencies with 3+ years of data: thin gray lines, low opacity
        - Top 6 flagged agencies (worst fit + most concern signals): red lines
        - Program-wide weighted mean: thick dark dashed reference line
        - Low-fit threshold at 3.5: subtle horizontal reference
        - Right-edge labels for each highlighted agency, with white outline
          so the text stays readable over gray background lines
    """
    # only agencies with at least 3 years of data — keeps the gray layer
    # meaningful and avoids one-year stubs cluttering the background
    year_counts = (
        trends_df[trends_df["data_quality"] == "sufficient"]
        .groupby("agency_name")["academic_year"]
        .count()
        .rename("year_count")
    )
    sufficient_agencies = year_counts[year_counts >= 3].index
    trends_sufficient = trends_df[trends_df["agency_name"].isin(sufficient_agencies)].copy()

    # top 6 flagged agencies: most concern signals first, then lowest fit score
    display_col = get_display_col(profiles_df)
    spotlight_df = (
        profiles_df[
            (profiles_df["concern_flag"] == "review recommended")
            & (profiles_df["agency_name"].isin(sufficient_agencies))
        ]
        .sort_values(
            ["concern_indicator_count", "composite_fit_score"],
            ascending=[False, True],
        )
        .head(6)
    )
    spotlight_agencies = set(spotlight_df["agency_name"])
    spotlight_labels = dict(zip(spotlight_df["agency_name"], spotlight_df[display_col]))

    # build a chronologically sorted x-axis from the year labels
    year_order = (
        trends_sufficient[["academic_year", "academic_year_start"]]
        .drop_duplicates()
        .sort_values("academic_year_start")["academic_year"]
        .tolist()
    )
    year_to_x = {year: i for i, year in enumerate(year_order)}

    # program-wide weighted mean per year — weights by response count so
    # agencies with more data contribute proportionally more to the average
    yearly_mean = (
        trends_sufficient.groupby("academic_year")
        .apply(
            lambda g: (g["composite_fit_score"] * g["response_count"]).sum()
            / g["response_count"].sum(),
            include_groups=False,
        )
        .round(2)
    )

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])

    # layer 1: all non-spotlight agencies in gray
    for agency_name, agency_df in trends_sufficient.groupby("agency_name"):
        if agency_name in spotlight_agencies:
            continue
        agency_sorted = (
            agency_df.set_index("academic_year").reindex(year_order).reset_index()
        )
        x_vals = [year_to_x[y] for y in agency_sorted["academic_year"]]
        y_vals = agency_sorted["composite_fit_score"].tolist()
        ax.plot(
            x_vals,
            y_vals,
            alpha=0.15,
            color=COLORS["neutral"],
            linewidth=0.9,
            solid_capstyle="round",
            zorder=1,
        )

    # layer 2: program mean as the reference anchor
    mean_x = [year_to_x[y] for y in year_order if y in yearly_mean.index]
    mean_y = [yearly_mean[y] for y in year_order if y in yearly_mean.index]
    ax.plot(
        mean_x,
        mean_y,
        alpha=0.9,
        color="#2c3e50",
        linestyle="--",
        linewidth=2.0,
        zorder=2,
        solid_capstyle="round",
    )

    # layer 3: spotlight agencies in slightly varying red shades
    # so overlapping lines at similar y-values stay distinguishable
    red_shades = [
        "#c0392b", "#e74c3c", "#a93226",
        "#cb4335", "#b03a2e", "#d98880",
    ]

    for i, (_, row) in enumerate(spotlight_df.iterrows()):
        agency_name = row["agency_name"]
        agency_df = trends_sufficient[trends_sufficient["agency_name"] == agency_name]
        # keep academic_year as the index so last_valid_index() returns a year
        # string like "2025-2026" rather than an integer position
        agency_sorted = agency_df.set_index("academic_year").reindex(year_order)
        scores = agency_sorted["composite_fit_score"]
        x_vals = [year_to_x[y] for y in agency_sorted.index]
        color = red_shades[i % len(red_shades)]

        ax.plot(
            x_vals,
            scores.tolist(),
            alpha=0.9,
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=4,
            markeredgewidth=0,
            solid_capstyle="round",
            zorder=4,
        )

        # right-edge label at the last non-null data point
        # white path-effect outline keeps the label readable over gray lines
        last_valid_idx = scores.last_valid_index()
        if last_valid_idx is not None:
            last_x = year_to_x[last_valid_idx]
            last_y = scores[last_valid_idx]
            raw_label = spotlight_labels.get(agency_name, agency_name)
            label = raw_label[:28] + "…" if len(raw_label) > 28 else raw_label
            ax.text(
                last_x + 0.12,
                last_y,
                label,
                fontsize=8,
                va="center",
                ha="left",
                color=color,
                fontweight="bold",
                path_effects=[
                    pe.withStroke(linewidth=2.5, foreground=COLORS["background"])
                ],
                zorder=5,
            )

    # low-fit reference line
    ax.axhline(3.5, alpha=0.4, color=COLORS["caution"], linestyle=":", linewidth=1.2, zorder=1)
    ax.text(
        0, 3.52,
        "low-fit threshold (3.5)",
        color=COLORS["caution"],
        fontsize=8,
        va="bottom",
        ha="left",
        alpha=0.8,
    )

    # axis formatting
    ax.set_xticks(range(len(year_order)))
    ax.set_xticklabels(year_order, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Placement Quality Score (1-5)", fontsize=10)
    ax.set_ylim(1.8, 5.2)
    ax.set_xlim(-0.4, len(year_order) - 0.4)
    ax.yaxis.grid(True, alpha=0.15, color=COLORS["neutral"], linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # legend
    legend_elements = [
        Line2D([0], [0], color=COLORS["neutral"], alpha=0.4, linewidth=1.5, label="All other agencies"),
        Line2D([0], [0], color="#2c3e50", linewidth=2.0, linestyle="--", label="Program mean"),
        Line2D([0], [0], color=COLORS["concern"], linewidth=2.2, label="Flagged agencies (highlighted)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8.5, frameon=False, loc="upper right")

    flagged_count = len(spotlight_agencies)
    total_count = trends_sufficient["agency_name"].nunique()
    ax.set_title(
        f"Figure 15. Among {total_count} agencies with multi-year trend data, "
        f"these {flagged_count} flagged agencies\n"
        "have the lowest placement quality scores — and most have been declining for several years",
        loc="left",
        fontsize=11,
        fontweight="bold",
        pad=14,
    )

    plt.tight_layout()
    save_figure(fig, "fig_15_agency_trend_spotlight.png")


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------


def generate_all_figures() -> None:
    """Build every static figure used by the capstone materials.

    Reads from the pipeline outputs. Run after pipeline.py finishes.
    """
    print("building report figures")

    profiles_df = pd.read_csv(profiles_file)
    sufficient_df = profiles_df[profiles_df["data_quality"] == "sufficient"].copy()
    row_df = pd.read_csv(input_file)
    trends_df = pd.read_csv(trends_file) if trends_file.exists() else None

    fig_01_concern_flag_summary(sufficient_df)
    print("  fig 01 done - concern flag summary")

    fig_02_fit_vs_recommendation(sufficient_df)
    print("  fig 02 done - fit vs recommendation scatter")

    fig_03_sentiment_distribution(sufficient_df)
    print("  fig 03 done - sentiment distribution")

    fig_04_likert_mean_scores(sufficient_df)
    print("  fig 04 done - likert mean scores (sorted)")

    fig_05_theme_frequency(sufficient_df)
    print("  fig 05 done - theme frequency (sorted by gap)")

    fig_06_bigrams_comparison(profiles_df)
    print("  fig 06 done - bigrams comparison")

    fig_07_wordcloud_most_helpful(row_df)
    print("  fig 07 done - word cloud most helpful")

    fig_08_wordcloud_least_helpful(row_df)
    print("  fig 08 done - word cloud least helpful")

    fig_09_top_flagged_agencies(sufficient_df)
    print("  fig 09 done - top flagged agencies")

    if trends_df is not None and not trends_df.empty:
        trends_df = trends_df.sort_values("academic_year_start")

        fig_10_yearly_fit_trend(trends_df)
        print("  fig 10 done - yearly fit trend (line only)")

        fig_11_yearly_evaluation_volume(trends_df)
        print("  fig 11 done - yearly evaluation volume (bar only)")

        fig_15_agency_trend_spotlight(trends_df, profiles_df)
        print("  fig 15 done - agency trend spotlight (gray-out)")
    else:
        print("  figs 10, 11, 15 skipped - trends file not found")

    fig_12_fit_score_distribution(sufficient_df)
    print("  fig 12 done - fit score distribution (eda)")

    fig_13_recommendation_rate_distribution(sufficient_df)
    print("  fig 13 done - recommendation rate distribution (eda)")

    fig_14_sentiment_score_distribution(sufficient_df)
    print("  fig 14 done - sentiment score distribution (eda)")

    print(f"done - {len(list(figures_dir.glob('*.png')))} figures saved to {figures_dir}")


if __name__ == "__main__":
    generate_all_figures()
