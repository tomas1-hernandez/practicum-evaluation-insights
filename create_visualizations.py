"""Build all static report figures for practicum evaluation intelligence.

Run after pipeline.py:
    python create_visualizations.py
"""

from __future__ import annotations

import re
from collections import Counter

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde, ttest_ind
from wordcloud import WordCloud

from config import (
    agency_profiles_file,
    agency_trends_file,
    concern_fit_score,
    concern_recommendation,
    concern_sentiment,
    evaluations_text_file,
    figure_dpi,
    figures_dir,
    input_file,
)
from constants import (
    likert_cols,
    likert_display,
    likert_map,
    placement_metric_labels_long,
    placement_quality_cols,
    program_level_display_map,
    stopwords,
)

sns.set_theme(style="ticks", font_scale=1.0)

# palette
WHITE = "#ffffff"
BLUE = "#5F89A3"
RED = "#B34F5B"
AMBER = "#A65C28"
TEAL = "#4A6F64"
LIGHT_BLUE = "#77A8CA"
MUTED_BLUE = "#B4CFE0"
GRAY_REF = "#869299"
GRAY_TEXT = "#4b555b"
GRAY_SPINE = "#c7cfd4"
BLUSH = "#f1e0e1"
TITLE = "#1c1e21"


# -----------------------------------------------------------------------------
# shared helpers
# -----------------------------------------------------------------------------
def _despine(ax):
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)


def _despine_left(ax):
    sns.despine(ax=ax, top=True, right=True, left=True, bottom=False)


def _spine_style(ax):
    for sp, spine in ax.spines.items():
        if sp in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_color(GRAY_SPINE)
            spine.set_linewidth(0.8)


def _base(ax, grid="y"):
    _spine_style(ax)
    ax.set_facecolor(WHITE)

    if grid == "y":
        ax.yaxis.grid(True, alpha=0.15, color=GRAY_SPINE)
        ax.xaxis.grid(False)
    elif grid == "x":
        ax.xaxis.grid(True, alpha=0.15, color=GRAY_SPINE)
        ax.yaxis.grid(False)
    else:
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)

    ax.set_axisbelow(True)
    ax.tick_params(colors=GRAY_TEXT, labelsize=8.5)


def _title(fig, title, subtitle=None):
    fig.text(
        0.01,
        0.99,
        title,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=TITLE,
    )
    if subtitle:
        fig.text(
            0.01,
            0.948,
            subtitle,
            ha="left",
            va="top",
            fontsize=9,
            color=GRAY_TEXT,
        )


def _save(fig, name):
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        figures_dir / name,
        dpi=figure_dpi,
        bbox_inches="tight",
        facecolor=WHITE,
        edgecolor="none",
    )
    plt.close(fig)


def _add_recommendation_pct(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    r = pd.to_numeric(df["recommendation_rate"], errors="coerce")
    df["rec_pct"] = r * 100 if r.dropna().max() <= 1.0 else r
    return df


def _parse_bigrams(series, top_n=15):
    artifacts = {
        "biggest help",
        "chances try",
        "day day",
        "day work",
        "enough hands",
        "hands practice",
        "helpful educational",
        "helpful least",
        "helpful part",
        "instead observing",
        "issues site",
        "least helpful",
        "most helpful",
        "needed chances",
        "needed client",
        "one frustration",
        "part least",
        "try skills",
        "work enough",
    }

    counter: Counter = Counter()
    for val in series.dropna():
        for item in str(val).split(" | ")[:top_n]:
            parts = item.rsplit(":", 1)
            if len(parts) == 2:
                counter[parts[0].strip()] += int(parts[1].strip())

    return Counter({k: v for k, v in counter.items() if k not in artifacts})


def _word_freq(series, extra_stop=None):
    stop = stopwords | (extra_stop or set())
    tokens: list[str] = []

    for text in series.dropna():
        words = re.findall(r"\b[a-z]+\b", str(text).lower())
        tokens.extend(w for w in words if w not in stop and len(w) > 2)

    return dict(Counter(tokens).most_common(90))


def _histogram(
    data,
    title,
    subtitle,
    xlabel,
    thresh_val,
    thresh_label,
    filename,
    x_lo=None,
    x_hi=None,
):
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    fig.patch.set_facecolor(WHITE)
    _base(ax, grid="y")

    ax.hist(
        data.dropna(),
        bins=22,
        color=LIGHT_BLUE,
        alpha=0.80,
        edgecolor=WHITE,
        linewidth=0.4,
    )

    ax.axvline(thresh_val, color=RED, linewidth=1.6, linestyle="--")
    ax.text(
        thresh_val + (data.max() - data.min()) * 0.01,
        ax.get_ylim()[1] * 0.90,
        thresh_label,
        color=RED,
        fontsize=8,
        va="top",
    )

    mean_val = data.mean()
    ax.axvline(mean_val, color=GRAY_REF, linewidth=1.0, linestyle=":")
    ax.text(
        mean_val + (data.max() - data.min()) * 0.01,
        ax.get_ylim()[1] * 0.70,
        f"Mean  {mean_val:.2f}",
        color=GRAY_TEXT,
        fontsize=8,
        va="top",
    )

    ax.set_xlabel(xlabel, fontsize=9, color=GRAY_TEXT, labelpad=6)
    ax.set_ylabel(
        "Number of agencies", fontsize=9, color=GRAY_TEXT, labelpad=6
    )
    ax.tick_params(colors=GRAY_TEXT, labelsize=8.5)

    if x_lo is not None:
        ax.set_xlim(x_lo, x_hi)

    _title(fig, title, subtitle)
    fig.subplots_adjust(top=0.87, bottom=0.13, left=0.08, right=0.97)
    _despine(ax)
    _save(fig, filename)


# -----------------------------------------------------------------------------
# figure 1 - concern flag summary
# -----------------------------------------------------------------------------
def fig_01_concern_flag_summary(sufficient: pd.DataFrame) -> None:
    total = len(sufficient)
    flagged = int(
        (sufficient["concern_flag"].str.lower() == "review recommended").sum()
    )
    not_flagged = total - flagged
    pct_flagged = round(flagged / total * 100, 1)
    pct_ok = round(100 - pct_flagged, 1)

    fig, ax = plt.subplots(figsize=(11, 2.8))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_yticks([])

    bar_h = 0.62

    ax.barh([""], [pct_ok], color=MUTED_BLUE, height=bar_h, left=0)
    ax.barh([""], [pct_flagged], color=RED, height=bar_h, left=pct_ok)

    ax.text(
        pct_ok / 2,
        0,
        f"{not_flagged} agencies\n{pct_ok:.1f}%",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=GRAY_TEXT,
        linespacing=1.4,
    )

    ax.text(
        pct_ok + pct_flagged / 2,
        0,
        f"{flagged} agencies\n{pct_flagged:.1f}%",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=WHITE,
        linespacing=1.4,
    )

    ax.set_xlim(0, 100)
    ax.set_xticks([])

    fig.text(
        0.01,
        0.99,
        "Nearly 3 in 10 practicum agencies met the threshold for leadership review",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=TITLE,
    )
    fig.text(
        0.01,
        0.90,
        "An agency is flagged when two or more concern signals occur at the same time.",
        ha="left",
        va="top",
        fontsize=9,
        color=GRAY_TEXT,
    )

    fig.subplots_adjust(top=0.72, bottom=0.08, left=0.01, right=0.99)
    _save(fig, "fig_01_concern_flag_summary.png")


# -----------------------------------------------------------------------------
# figure 2 - fit vs recommendation
# -----------------------------------------------------------------------------
def fig_02_fit_vs_recommendation(suf: pd.DataFrame) -> None:
    suf = _add_recommendation_pct(suf)
    mask = suf["concern_flag"].str.lower() == "review recommended"

    fig, ax = plt.subplots(figsize=(10, 6.8))
    fig.patch.set_facecolor(WHITE)
    _base(ax, grid="none")

    ax.add_patch(
        mpatches.Rectangle(
            (1.8, 0),
            concern_fit_score - 1.8,
            concern_recommendation * 100,
            facecolor=BLUSH,
            edgecolor="none",
            alpha=0.65,
            zorder=0,
        )
    )

    ax.text(
        1.95,
        60,
        "Both signals weak",
        color="#be554d",
        fontsize=8,
        va="top",
        ha="left",
        linespacing=1.35,
        fontstyle="italic",
    )

    ax.axhline(
        concern_recommendation * 100,
        color=GRAY_REF,
        linewidth=0.9,
        linestyle="--",
        zorder=1,
    )
    ax.axvline(
        concern_fit_score,
        color=GRAY_REF,
        linewidth=0.9,
        linestyle=":",
        zorder=1,
    )

    ax.text(
        4.96,
        concern_recommendation * 100 + 1.2,
        f"{int(concern_recommendation * 100)}% threshold",
        color=GRAY_REF,
        fontsize=7.5,
        ha="right",
        va="bottom",
    )
    ax.text(
        concern_fit_score + 0.04,
        2,
        f"{concern_fit_score:.1f} threshold",
        color=GRAY_REF,
        fontsize=7.5,
        ha="left",
        va="bottom",
    )

    ax.scatter(
        suf.loc[~mask, "placement_quality_score"],
        suf.loc[~mask, "rec_pct"],
        color=MUTED_BLUE,
        s=30,
        alpha=0.55,
        linewidths=0,
        zorder=2,
    )
    ax.scatter(
        suf.loc[mask, "placement_quality_score"],
        suf.loc[mask, "rec_pct"],
        color=RED,
        s=42,
        alpha=0.80,
        marker="^",
        edgecolors=WHITE,
        linewidths=0.4,
        zorder=4,
    )

    ax.set_xlim(1.8, 5.05)
    ax.set_ylim(0, 104)
    ax.set_xlabel(
        "Placement Quality Score", fontsize=9, color=GRAY_TEXT, labelpad=6
    )
    ax.set_ylabel(
        "Recommendation rate (%)", fontsize=9, color=GRAY_TEXT, labelpad=6
    )

    title = "Flagged agencies cluster where placement quality and recommendation rates are both low"
    subtitle = (
        "Red triangles mark agencies needing review. Gray dots show unflagged agencies. "
        "The shaded area marks where both measures fall below the concern thresholds."
    )
    _title(fig, title, subtitle)

    fig.subplots_adjust(top=0.87, bottom=0.11, left=0.09, right=0.97)
    _despine(ax)
    _save(fig, "fig_02_fit_vs_recommendation.png")


# -----------------------------------------------------------------------------
# figure 3 - sentiment distribution
# -----------------------------------------------------------------------------
def fig_03_sentiment_distribution(suf: pd.DataFrame) -> None:
    flagged = suf[suf["concern_flag"].str.lower() == "review recommended"][
        "overall_sentiment_score"
    ].dropna()
    not_flagged = suf[suf["concern_flag"].str.lower() == "no flag"][
        "overall_sentiment_score"
    ].dropna()

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    fig.patch.set_facecolor(WHITE)
    _base(ax, grid="none")
    ax.set_yticks([])

    sns.kdeplot(
        not_flagged,
        ax=ax,
        color=GRAY_REF,
        fill=True,
        alpha=0.45,
        linewidth=1.5,
    )
    sns.kdeplot(
        flagged,
        ax=ax,
        color=RED,
        fill=True,
        alpha=0.50,
        linewidth=1.8,
    )

    grid = np.linspace(
        min(flagged.min(), not_flagged.min()),
        max(flagged.max(), not_flagged.max()),
        400,
    )

    for series, label, color in [
        (not_flagged, "no flag", GRAY_REF),
        (flagged, "flagged", RED),
    ]:
        density = gaussian_kde(series)(grid)
        peak_x = grid[density.argmax()]
        peak_y = density.max()
        ax.text(
            peak_x,
            peak_y * 1.05,
            label,
            color=color,
            fontsize=9.5,
            fontweight="bold",
            ha="center",
            va="bottom",
        )

    ax.axvline(
        concern_sentiment,
        color=GRAY_REF,
        linewidth=1.0,
        linestyle=":",
        zorder=1,
    )
    ax.text(
        0.07,
        0.94,
        f"Concern threshold  {concern_sentiment:.2f}",
        transform=ax.transAxes,
        color=GRAY_REF,
        fontsize=7.5,
        va="top",
    )

    ax.set_xlabel(
        "Overall sentiment score (VADER)",
        fontsize=9,
        color=GRAY_TEXT,
        labelpad=6,
    )

    title = "Flagged agencies skew toward lower sentiment, but there is considerable overlap between the two groups"
    subtitle = "Sentiment is one of five key concern signals, which is why it should be used with other indicators, not alone."
    _title(fig, title, subtitle)

    fig.subplots_adjust(top=0.87, bottom=0.13, left=0.03, right=0.97)
    _despine_left(ax)
    _save(fig, "fig_03_sentiment_distribution.png")


# -----------------------------------------------------------------------------
# figure 4 - likert mean scores
# -----------------------------------------------------------------------------
def fig_04_likert_mean_scores(suf: pd.DataFrame) -> None:
    means = {
        col: pd.to_numeric(suf[col], errors="coerce").mean()
        for col in likert_cols
    }

    plot_df = (
        pd.DataFrame.from_dict(means, orient="index", columns=["mean"])
        .dropna()
        .sort_values("mean", ascending=True)
        .reset_index()
        .rename(columns={"index": "col"})
    )
    plot_df["label"] = plot_df["col"].map(likert_display)
    n = len(plot_df)

    colors = [RED if i < 3 else MUTED_BLUE for i in range(n)]
    mean_all = plot_df["mean"].mean()

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    fig.patch.set_facecolor(WHITE)
    _base(ax, grid="none")
    ax.tick_params(left=False)
    ax.xaxis.grid(True, alpha=0.15, color=GRAY_SPINE)

    ax.barh(range(n), plot_df["mean"], color=colors, height=0.55, alpha=0.88)
    ax.axvline(
        mean_all, color=GRAY_REF, linewidth=1.0, linestyle="--", alpha=0.7
    )
    ax.text(
        mean_all + 0.02,
        n - 0.3,
        f"Mean  {mean_all:.2f}",
        color=GRAY_REF,
        fontsize=8,
        va="top",
    )

    ax.set_yticks(range(n))
    ax.set_yticklabels(plot_df["label"], fontsize=9.5, color="#3d4551")
    ax.set_xlim(1, 5.3)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(
        ["1", "2", "3", "4", "5"], fontsize=8.5, color=GRAY_TEXT
    )
    ax.set_xlabel(
        "Mean score (1 = strongly disagree, 5 = strongly agree)",
        fontsize=9,
        color=GRAY_TEXT,
        labelpad=6,
    )

    title = "Students are satisfied with supervision but feel less ready for independent practice"
    subtitle = "The red bars indicate the three lowest-scoring items related to practice readiness, listed from highest to lowest."
    _title(fig, title, subtitle)

    fig.subplots_adjust(top=0.87, bottom=0.11, left=0.33, right=0.97)
    _despine_left(ax)
    _save(fig, "fig_04_likert_mean_scores.png")


# -----------------------------------------------------------------------------
# figure 5 - theme frequency
# -----------------------------------------------------------------------------
def fig_05_theme_frequency(suf: pd.DataFrame) -> None:
    theme_display = {
        "administrative_overload": "Administrative overload",
        "direct_practice_opportunity": "Direct practice opportunity",
        "learning_environment": "Learning environment",
        "organizational_structure": "Organizational structure",
        "social_justice_alignment": "Social justice alignment",
        "strong_supervision": "Strong supervision",
    }

    themes = sorted(theme_display)
    rows = []
    for theme in themes:
        rows.append(
            {
                "theme": theme,
                "label": theme_display[theme],
                "helpful": pd.to_numeric(
                    suf[f"{theme}_helpful_pct"], errors="coerce"
                ).mean(),
                "least": pd.to_numeric(
                    suf[f"{theme}_least_pct"], errors="coerce"
                ).mean(),
            }
        )

    plot_df = (
        pd.DataFrame(rows)
        .assign(abs_gap=lambda d: (d["least"] - d["helpful"]).abs())
        .sort_values("abs_gap", ascending=True)
        .reset_index(drop=True)
    )
    n = len(plot_df)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    fig.patch.set_facecolor(WHITE)
    _base(ax, grid="none")
    ax.xaxis.grid(True, alpha=0.15, color=GRAY_SPINE)
    ax.tick_params(left=False)

    for y, (_, row) in enumerate(plot_df.iterrows()):
        ax.plot(
            [row["helpful"], row["least"]],
            [y, y],
            color=GRAY_SPINE,
            linewidth=1.6,
            zorder=2,
            solid_capstyle="round",
        )
        ax.scatter(row["helpful"], y, color=BLUE, s=72, linewidths=0, zorder=4)
        ax.scatter(row["least"], y, color=RED, s=72, linewidths=0, zorder=4)

    ax.set_yticks(range(n))
    ax.set_yticklabels(plot_df["label"], fontsize=10, color="#3d4551")
    ax.set_xlabel(
        "Mean % of agency responses tagged with this theme",
        fontsize=9,
        color=GRAY_TEXT,
        labelpad=6,
    )
    ax.set_xlim(left=0)

    title = (
        "Strong placements rely on strong supervision and practice,"
        "while weak ones suffer from excessive paperwork and little client contact"
    )
    subtitle = (
        "Blue dot = appeared more often in most-helpful comments      "
        "Red dot = appeared more often in least-helpful comments"
    )
    _title(fig, title, subtitle)

    fig.subplots_adjust(top=0.84, bottom=0.13, left=0.25, right=0.97)
    _despine(ax)
    _save(fig, "fig_05_theme_frequency.png")


# -----------------------------------------------------------------------------
# figure 6 - bigram comparison
# -----------------------------------------------------------------------------
def fig_06_bigrams_comparison(profiles_df: pd.DataFrame) -> None:
    helpful = _parse_bigrams(profiles_df["most_helpful_top_bigrams"])
    least = _parse_bigrams(profiles_df["least_helpful_top_bigrams"])

    top_h = helpful.most_common(12)
    top_l = least.most_common(12)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.5, 5.8))
    fig.patch.set_facecolor(WHITE)

    for ax, data, color, panel_title in [
        (ax_l, top_h, BLUE, "Most-helpful comments - top phrases"),
        (ax_r, top_l, RED, "Least-helpful comments - top phrases"),
    ]:
        ax.set_facecolor(WHITE)
        ax.barh(
            range(len(data)),
            [count for _, count in data],
            color=color,
            alpha=0.85,
            height=0.6,
        )
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels([phrase for phrase, _ in data], fontsize=9)
        ax.invert_yaxis()
        _spine_style(ax)
        ax.xaxis.grid(True, alpha=0.15, color=GRAY_SPINE)
        ax.yaxis.grid(False)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", colors=GRAY_TEXT, labelsize=8.5)
        ax.tick_params(left=False)
        ax.set_xlabel(
            "Cumulative mentions across agencies",
            fontsize=8.5,
            color=GRAY_TEXT,
            labelpad=5,
        )
        ax.set_title(
            panel_title,
            loc="left",
            fontsize=10,
            fontweight="bold",
            color=color,
            pad=10,
        )
        sns.despine(ax=ax, top=True, right=True, left=True, bottom=False)

    title = "Supportive language shows growth, while unhelpful language reveals frustration"
    subtitle = "The leading two-word phrases exclude phrases that simply repeat the survey prompt wording."
    _title(fig, title, subtitle)

    fig.subplots_adjust(
        top=0.87, bottom=0.11, left=0.10, right=0.98, wspace=0.45
    )
    _save(fig, "fig_06_bigrams_comparison.png")


# -----------------------------------------------------------------------------
# figure 7 - word cloud - most helpful
# -----------------------------------------------------------------------------
def fig_07_wordcloud_most_helpful(row_df: pd.DataFrame) -> None:
    freq = _word_freq(row_df["most_helpful"])
    top12 = set(list(freq)[:12])

    def color_func(word, **kwargs):
        return BLUE if word in top12 else "#c8d3da"

    wc = WordCloud(
        background_color=WHITE,
        color_func=color_func,
        width=1400,
        height=680,
        max_words=80,
        prefer_horizontal=0.85,
        collocations=False,
        min_font_size=11,
        max_font_size=160,
    ).generate_from_frequencies(freq)

    fig, ax = plt.subplots(figsize=(12, 5.8))
    fig.patch.set_facecolor(WHITE)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    _save(fig, "fig_07_wordcloud_most_helpful.png")


# -----------------------------------------------------------------------------
# figure 8 - word cloud - least helpful
# -----------------------------------------------------------------------------
def fig_08_wordcloud_least_helpful(row_df: pd.DataFrame) -> None:
    extra = {
        "one",
        "thing",
        "would",
        "also",
        "really",
        "lot",
        "could",
        "much",
        "even",
        "sometimes",
        "though",
    }
    freq = _word_freq(row_df["least_helpful"], extra_stop=extra)
    top12 = set(list(freq)[:12])

    def color_func(word, **kwargs):
        return RED if word in top12 else "#cfc8c8"

    wc = WordCloud(
        background_color=WHITE,
        color_func=color_func,
        width=1400,
        height=680,
        max_words=80,
        prefer_horizontal=0.85,
        collocations=False,
        min_font_size=11,
        max_font_size=160,
    ).generate_from_frequencies(freq)

    fig, ax = plt.subplots(figsize=(12, 5.8))
    fig.patch.set_facecolor(WHITE)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    _save(fig, "fig_08_wordcloud_least_helpful.png")


# -----------------------------------------------------------------------------
# figure 9 - top flagged agencies
# -----------------------------------------------------------------------------
def fig_09_top_flagged_agencies(suf: pd.DataFrame) -> None:
    flagged = (
        suf[suf["concern_flag"].str.lower() == "review recommended"]
        .sort_values("placement_quality_score", ascending=True)
        .head(10)
        .copy()
    )
    flagged["label"] = flagged["agency_name_display"].apply(
        lambda s: s if len(s) <= 40 else s[:39].rstrip() + "…"
    )
    all_mean = suf["placement_quality_score"].mean()

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    fig.patch.set_facecolor(WHITE)
    _base(ax, grid="none")
    ax.xaxis.grid(True, alpha=0.15, color=GRAY_SPINE)
    ax.tick_params(left=False)

    y_pos = list(range(len(flagged)))
    ax.barh(
        y_pos,
        flagged["placement_quality_score"],
        color=RED,
        alpha=0.85,
        height=0.55,
    )
    ax.invert_yaxis()

    ax.axvline(
        all_mean, color=GRAY_REF, linewidth=1.0, linestyle="--", alpha=0.8
    )
    ax.text(
        all_mean + 0.02,
        len(flagged) - 0.5,
        f"All-agency mean  {all_mean:.2f}",
        color=GRAY_REF,
        fontsize=8,
        va="top",
    )

    ax.axvline(
        concern_fit_score,
        color=GRAY_REF,
        linewidth=1.0,
        linestyle=":",
        alpha=0.8,
    )
    ax.text(
        concern_fit_score + 0.02,
        -0.4,
        f"Concern threshold {concern_fit_score:.1f}",
        color=GRAY_REF,
        fontsize=7.5,
        va="top",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(flagged["label"], fontsize=9)
    ax.set_xlim(0, 4.4)
    ax.set_xlabel(
        "Placement Quality Score", fontsize=9, color=GRAY_TEXT, labelpad=6
    )
    ax.tick_params(axis="x", colors=GRAY_TEXT, labelsize=8.5)

    title = "The lowest-scoring flagged agencies fall well below the all-agency average on Placement Quality Score"
    subtitle = "These are the ten flagged agencies with the lowest Placement Quality Scores."
    _title(fig, title, subtitle)

    fig.subplots_adjust(top=0.87, bottom=0.11, left=0.38, right=0.97)
    _despine_left(ax)
    _save(fig, "fig_09_top_flagged_agencies.png")


# -----------------------------------------------------------------------------
# figure 10 - yearly fit trend
# -----------------------------------------------------------------------------
def fig_10_yearly_fit_trend(trends_df: pd.DataFrame) -> None:
    trends_df = trends_df.copy()
    for col in [
        "placement_quality_score",
        "response_count",
        "academic_year_start",
    ]:
        trends_df[col] = pd.to_numeric(trends_df[col], errors="coerce")

    yearly = (
        trends_df.groupby("academic_year")
        .apply(
            lambda g: pd.Series(
                {
                    "start": g["academic_year_start"].iloc[0],
                    "mean_score": round(
                        (
                            g["placement_quality_score"] * g["response_count"]
                        ).sum()
                        / g["response_count"].sum(),
                        2,
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
        .sort_values("start")
        .reset_index(drop=True)
    )

    years = yearly["academic_year"].tolist()
    scores = yearly["mean_score"].tolist()
    x = list(range(len(years)))

    fig, ax = plt.subplots(figsize=(12, 5.2))
    fig.patch.set_facecolor(WHITE)
    _base(ax, grid="none")
    ax.yaxis.grid(True, alpha=0.15, color=GRAY_SPINE)

    ax.axhline(
        concern_fit_score,
        color=GRAY_REF,
        linewidth=0.9,
        linestyle=":",
        zorder=1,
    )
    ax.text(
        0.02,
        concern_fit_score + 0.02,
        f"Concern threshold  {concern_fit_score:.1f}",
        color=GRAY_REF,
        fontsize=8,
        va="bottom",
    )

    ax.plot(
        x, scores, color=BLUE, linewidth=2.2, solid_capstyle="round", zorder=3
    )
    ax.scatter(x[-1], scores[-1], color=RED, s=80, zorder=5, linewidths=0)

    ax.annotate(
        f"{scores[-1]:.2f}\nFirst time below\nthe concern threshold",
        xy=(x[-1], scores[-1]),
        xytext=(-3, -0.28),
        textcoords="offset fontsize",
        fontsize=8.5,
        color=RED,
        fontweight="bold",
        ha="right",
        va="top",
        linespacing=1.35,
        path_effects=[pe.withStroke(linewidth=2.5, foreground=WHITE)],
        arrowprops=dict(arrowstyle="-", color=RED, lw=0.8),
        zorder=6,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        years, rotation=38, ha="right", fontsize=8.5, color=GRAY_TEXT
    )
    ax.set_ylim(3.0, 4.7)
    ax.set_yticks([3.0, 3.5, 4.0, 4.5])
    ax.set_yticklabels(
        ["3.0", "3.5", "4.0", "4.5"], fontsize=8.5, color=GRAY_TEXT
    )
    ax.set_ylabel(
        "Placement Quality Score", fontsize=9, color=GRAY_TEXT, labelpad=6
    )

    title = "In 2025-2026, the overall Placement Quality Score fell below the concern threshold for the first time"
    subtitle = "This shows the yearly average Placement Quality Score, weighted by the number of responses."
    _title(fig, title, subtitle)

    fig.subplots_adjust(top=0.87, bottom=0.16, left=0.07, right=0.97)
    _despine(ax)
    _save(fig, "fig_10_yearly_fit_trend.png")


# -----------------------------------------------------------------------------
# figure 11 - yearly evaluation volume
# -----------------------------------------------------------------------------
def fig_11_yearly_evaluation_volume(trends_df: pd.DataFrame) -> None:
    trends_df = trends_df.copy()
    for col in ["academic_year_start", "response_count"]:
        trends_df[col] = pd.to_numeric(trends_df[col], errors="coerce")

    yearly = (
        trends_df.groupby("academic_year")
        .apply(
            lambda g: pd.Series(
                {
                    "start": g["academic_year_start"].iloc[0],
                    "count": int(g["response_count"].sum()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
        .sort_values("start")
    )

    fig, ax = plt.subplots(figsize=(11, 4.8))
    fig.patch.set_facecolor(WHITE)
    _base(ax, grid="y")

    ax.bar(
        range(len(yearly)),
        yearly["count"],
        color=BLUE,
        alpha=0.90,
        width=0.65,
    )
    ax.set_xticks(range(len(yearly)))
    ax.set_xticklabels(
        yearly["academic_year"],
        rotation=38,
        ha="right",
        fontsize=8.5,
        color=GRAY_TEXT,
    )
    ax.set_ylabel(
        "Number of evaluations submitted",
        fontsize=9,
        color=GRAY_TEXT,
        labelpad=6,
    )
    ax.tick_params(axis="y", colors=GRAY_TEXT, labelsize=8.5)

    title = "Evaluation collection more than doubled after 2020, creating a richer evidence base for recent years"
    subtitle = "Volume was flat from 2014 to 2021, then grew sharply. More responses per year means more reliable agency profiles."
    _title(fig, title, subtitle)

    fig.subplots_adjust(top=0.87, bottom=0.18, left=0.08, right=0.97)
    _despine(ax)
    _save(fig, "fig_11_yearly_evaluation_volume.png")


# -----------------------------------------------------------------------------
# figure 12 - fit score distribution
# -----------------------------------------------------------------------------
def fig_12_fit_score_distribution(suf: pd.DataFrame) -> None:
    data = suf["placement_quality_score"].dropna()
    _histogram(
        data,
        "Most agencies score well; the lower tail is where the concern flag clusters",
        "Distribution of Placement Quality Scores.",
        "Placement Quality Score (1 to 5 scale)",
        concern_fit_score,
        f"Concern threshold  {concern_fit_score:.1f}",
        "fig_12_fit_score_distribution.png",
        x_lo=1.8,
        x_hi=5.2,
    )


# -----------------------------------------------------------------------------
# figure 13 - recommendation rate distribution
# -----------------------------------------------------------------------------
def fig_13_recommendation_rate_distribution(suf: pd.DataFrame) -> None:
    suf = _add_recommendation_pct(suf)
    data = suf["rec_pct"].dropna()
    _histogram(
        data,
        "Most agencies are recommended by most students; a smaller group falls below the threshold",
        "Distribution of recommendation rates.",
        "Recommendation rate (%)",
        concern_recommendation * 100,
        f"Review threshold  {int(concern_recommendation * 100)}%",
        "fig_13_recommendation_rate_distribution.png",
    )


# -----------------------------------------------------------------------------
# figure 14 - sentiment distribution by trend
# -----------------------------------------------------------------------------
def fig_14_sentiment_by_trend(suf: pd.DataFrame) -> None:
    suf = suf.copy()
    suf["overall_sentiment_score"] = pd.to_numeric(
        suf["overall_sentiment_score"], errors="coerce"
    )

    trend_map = {"declining": RED, "stable": GRAY_TEXT, "improving": BLUE}

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    fig.patch.set_facecolor(WHITE)
    _base(ax, grid="none")
    ax.set_yticks([])

    peaks = []
    for trend, color in trend_map.items():
        subset = suf[suf["fit_trend"].str.lower() == trend][
            "overall_sentiment_score"
        ].dropna()
        if len(subset) < 5:
            continue

        sns.kdeplot(
            subset,
            ax=ax,
            color=color,
            fill=True,
            alpha=0.45,
            linewidth=1.6,
        )

        grid = np.linspace(subset.min(), subset.max(), 400)
        density = gaussian_kde(subset)(grid)
        peaks.append((trend, color, grid[density.argmax()], density.max()))

    for trend, color, peak_x, peak_y in peaks:
        ax.text(
            peak_x,
            peak_y * 1.05,
            trend,
            color=color,
            fontsize=9.5,
            fontweight="bold",
            ha="center",
            va="bottom",
        )

    ax.axvline(
        concern_sentiment,
        color=GRAY_REF,
        linewidth=0.9,
        linestyle=":",
        zorder=1,
    )
    ax.text(
        0.07,
        0.94,
        f"Concern threshold  {concern_sentiment:.2f}",
        transform=ax.transAxes,
        color=GRAY_REF,
        fontsize=7.5,
        va="top",
    )
    ax.set_xlabel(
        "Overall sentiment score (VADER)",
        fontsize=9,
        color=GRAY_TEXT,
        labelpad=6,
    )

    title = "Sentiment scores are mildly positive for most agencies, but declining agencies skew toward lower sentiment"
    subtitle = "Sentiment distribution by placement quality trend direction."
    _title(fig, title, subtitle)

    fig.subplots_adjust(top=0.87, bottom=0.13, left=0.03, right=0.97)
    _despine_left(ax)
    _save(fig, "fig_14_sentiment_distribution_eda.png")


# -----------------------------------------------------------------------------
# figure 15 - agency trend spotlight
# -----------------------------------------------------------------------------
def fig_15_agency_trend_spotlight(
    trends_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
) -> None:
    trends_df = trends_df.copy()
    for col in [
        "placement_quality_score",
        "response_count",
        "academic_year_start",
    ]:
        trends_df[col] = pd.to_numeric(trends_df[col], errors="coerce")

    year_counts = (
        trends_df[trends_df["data_quality"] == "sufficient"]
        .groupby("agency_name")["academic_year"]
        .count()
    )
    sufficient = set(year_counts[year_counts >= 3].index)
    ts = trends_df[trends_df["agency_name"].isin(sufficient)].copy()

    spot = (
        profiles_df[
            profiles_df["concern_flag"].str.lower().eq("review recommended")
            & profiles_df["agency_name"].isin(sufficient)
        ]
        .sort_values(
            ["concern_indicator_count", "placement_quality_score"],
            ascending=[False, True],
        )
        .head(3)
    )

    spotlight_names = list(spot["agency_name"])
    spotlight_labels = dict(
        zip(spot["agency_name"], spot["agency_name_display"])
    )
    warm = ["#5B252B", "#B34F5B", "#C5979B"]
    agency_color = {name: warm[i] for i, name in enumerate(spotlight_names)}

    year_order = (
        ts[["academic_year", "academic_year_start"]]
        .drop_duplicates()
        .sort_values("academic_year_start")["academic_year"]
        .tolist()
    )
    year_to_x = {year: i for i, year in enumerate(year_order)}
    n_years = len(year_order)

    yearly_mean = (
        ts.groupby("academic_year")
        .apply(
            lambda g: pd.Series(
                {
                    "start": g["academic_year_start"].iloc[0],
                    "mean": (
                        g["placement_quality_score"] * g["response_count"]
                    ).sum()
                    / g["response_count"].sum(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
        .sort_values("start")
    )

    fig, ax = plt.subplots(figsize=(12, 6.2))
    fig.patch.set_facecolor(WHITE)
    _base(ax, grid="none")

    for agency, group in ts.groupby("agency_name"):
        if agency in spotlight_names:
            continue

        group = group.sort_values("academic_year_start")
        xs = [year_to_x[year] for year in group["academic_year"]]
        ys = [
            group.loc[
                group["academic_year"] == year, "placement_quality_score"
            ].iloc[0]
            for year in group["academic_year"]
        ]

        if len(xs) >= 2:
            ax.plot(
                xs,
                ys,
                color=MUTED_BLUE,
                linewidth=0.7,
                alpha=0.3,
                solid_capstyle="round",
                zorder=1,
            )

    mean_xs = [year_to_x[year] for year in yearly_mean["academic_year"]]
    mean_ys = [
        yearly_mean.loc[yearly_mean["academic_year"] == year, "mean"].iloc[0]
        for year in yearly_mean["academic_year"]
    ]
    ax.plot(
        mean_xs,
        mean_ys,
        color=GRAY_TEXT,
        linewidth=1.6,
        linestyle="--",
        solid_capstyle="round",
        zorder=2,
    )

    ax.axhline(
        concern_fit_score,
        color=GRAY_REF,
        linewidth=0.9,
        linestyle=":",
        zorder=1,
    )
    ax.text(
        0.05,
        concern_fit_score + 0.03,
        f"Concern threshold  {concern_fit_score:.1f}",
        color=GRAY_REF,
        fontsize=8,
        va="bottom",
    )

    label_items = []
    if mean_xs:
        label_items.append(
            {
                "y": mean_ys[-1],
                "text": "Agency mean",
                "color": GRAY_TEXT,
                "fontweight": "normal",
                "fontsize": 8,
                "stroke": False,
            }
        )

    for name in spotlight_names:
        color = agency_color[name]
        group = ts[ts["agency_name"] == name].sort_values(
            "academic_year_start"
        )
        xs = [year_to_x[year] for year in group["academic_year"]]
        ys = [
            group.loc[
                group["academic_year"] == year, "placement_quality_score"
            ].iloc[0]
            for year in group["academic_year"]
        ]

        if len(xs) < 2:
            continue

        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=2.4,
            solid_capstyle="round",
            zorder=4,
        )

        raw = spotlight_labels[name]
        label = raw[:30] + "…" if len(raw) > 30 else raw
        label_items.append(
            {
                "y": ys[-1],
                "text": label,
                "color": color,
                "fontweight": "bold",
                "fontsize": 8.8,
                "stroke": True,
            }
        )

    label_items.sort(key=lambda d: -d["y"])
    min_gap = 0.22
    placed_y = None
    x_right = (n_years - 1) + 0.18

    for item in label_items:
        y_placed = (
            item["y"]
            if placed_y is None
            else min(item["y"], placed_y - min_gap)
        )
        placed_y = y_placed

        kwargs = {
            "color": item["color"],
            "fontsize": item["fontsize"],
            "va": "center",
            "ha": "left",
            "fontweight": item["fontweight"],
            "zorder": 5,
        }
        if item["stroke"]:
            kwargs["path_effects"] = [
                pe.withStroke(linewidth=2.5, foreground=WHITE)
            ]

        ax.text(x_right, y_placed, item["text"], **kwargs)

    ax.set_xticks(range(n_years))
    ax.set_xticklabels(
        year_order,
        rotation=38,
        ha="right",
        fontsize=8.5,
        color=GRAY_TEXT,
    )
    ax.set_ylim(1.3, 5.3)
    ax.set_yticks([2, 3, 4, 5])
    ax.set_yticklabels(["2", "3", "4", "5"], fontsize=8.5, color=GRAY_TEXT)
    ax.set_xlim(-0.4, n_years + 1.8)
    ax.set_ylabel(
        "Placement Quality Score", fontsize=9, color=GRAY_TEXT, labelpad=6
    )

    title = (
        "Trend tracking highlights agencies that have stayed weak over time"
    )
    subtitle = "Gray lines show all agencies. Highlighted agencies are flagged sites with the weakest recent multi-year trajectories."
    _title(fig, title, subtitle)

    fig.subplots_adjust(top=0.87, bottom=0.15, left=0.05, right=0.80)
    _despine(ax)
    _save(fig, "fig_15_agency_trend_spotlight.png")


# -----------------------------------------------------------------------------
# figure 16A - BSW vs MSW dumbbell
# -----------------------------------------------------------------------------
def fig_16a_bsw_msw_dumbbell(raw_df: pd.DataFrame) -> None:
    bsw_color = "#548578"
    msw_color = "#94555c"
    alpha_sig = 1.00
    alpha_ns = 0.28

    df = raw_df.copy()
    df["program_level"] = (
        df["program_year"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(program_level_display_map)
    )

    for col in placement_quality_cols:
        df[col] = df[col].astype(str).str.strip().str.lower().map(likert_map)

    df["Placement Quality Score"] = df[placement_quality_cols].mean(axis=1)

    bsw = df[df["program_level"] == "BSW"]
    msw = df[df["program_level"] == "MSW"]

    rows = []
    for col, label in placement_metric_labels_long.items():
        a = bsw[col].dropna()
        b = msw[col].dropna()
        _, p_value = ttest_ind(a, b, equal_var=False)

        rows.append(
            {
                "label": label,
                "bsw_mean": round(a.mean(), 3),
                "msw_mean": round(b.mean(), 3),
                "gap": round(b.mean() - a.mean(), 3),
                "significant": p_value < 0.05,
            }
        )

    plot_df = (
        pd.DataFrame(rows)
        .sort_values("gap", ascending=True)
        .reset_index(drop=True)
    )
    n = len(plot_df)
    n_sig = int(plot_df["significant"].sum())
    x_min = plot_df[["bsw_mean", "msw_mean"]].min().min() - 0.10
    x_max = plot_df[["bsw_mean", "msw_mean"]].max().max() + 0.20

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    fig.patch.set_facecolor(WHITE)
    _base(ax, grid="none")
    ax.xaxis.grid(True, alpha=0.15, color=GRAY_SPINE)
    ax.tick_params(left=False)

    for y, (_, row) in enumerate(plot_df.iterrows()):
        point_alpha = alpha_sig if row["significant"] else alpha_ns
        line_alpha = 0.55 if row["significant"] else 0.18

        ax.plot(
            [row["bsw_mean"], row["msw_mean"]],
            [y, y],
            color=GRAY_SPINE,
            linewidth=1.6,
            alpha=line_alpha,
            zorder=2,
            solid_capstyle="round",
        )
        ax.scatter(
            row["bsw_mean"],
            y,
            color=bsw_color,
            s=74,
            alpha=point_alpha,
            linewidths=0,
            zorder=4,
        )
        ax.scatter(
            row["msw_mean"],
            y,
            color=msw_color,
            s=74,
            alpha=point_alpha,
            linewidths=0,
            zorder=4,
        )

    ax.text(
        x_max,
        n - 0.45,
        "BSW",
        color=bsw_color,
        fontsize=8.5,
        ha="right",
        va="center",
        fontweight="bold",
    )
    ax.text(
        x_max,
        n - 1.0,
        "MSW",
        color=msw_color,
        fontsize=8.5,
        ha="right",
        va="center",
        fontweight="bold",
    )

    ax.set_yticks(range(n))
    ax.set_yticklabels(plot_df["label"], fontsize=9.5, color="#3d4551")
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Mean score", fontsize=9, color=GRAY_TEXT, labelpad=6)
    ax.tick_params(axis="x", colors=GRAY_TEXT, labelsize=8.5)

    title = "MSW students rated placements slightly higher than BSW students on all six measures"
    subtitle = (
        f"Full-color rows mark the {n_sig} measures where the difference was statistically meaningful. "
        "Muted rows show smaller differences not confirmed at the 0.05 level."
    )
    _title(fig, title, subtitle)

    fig.subplots_adjust(top=0.87, bottom=0.12, left=0.33, right=0.97)
    _despine_left(ax)
    _save(fig, "fig_16a_bsw_msw_dumbbell.png")


# -----------------------------------------------------------------------------
# figure 16B - recommendation rate bar
# -----------------------------------------------------------------------------
def fig_16b_bsw_msw_recommendation(raw_df: pd.DataFrame) -> None:
    bsw_color = "#42675d"
    msw_color = "#93424b"

    df = raw_df.copy()
    df["program_level"] = (
        df["program_year"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(program_level_display_map)
    )
    df["recommend_num"] = (
        df["recommend"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )

    bsw = df[df["program_level"] == "BSW"]
    msw = df[df["program_level"] == "MSW"]
    bsw_r = round(bsw["recommend_num"].mean() * 100, 1)
    msw_r = round(msw["recommend_num"].mean() * 100, 1)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    fig.patch.set_facecolor(WHITE)
    _base(ax, grid="y")

    ax.bar(
        ["BSW", "MSW"],
        [bsw_r, msw_r],
        color=[bsw_color, msw_color],
        width=0.4,
        alpha=0.88,
    )

    ax.set_ylim(0, min(100, max(bsw_r, msw_r) + 18))
    ax.set_ylabel(
        "Recommendation rate (%)", fontsize=9, color=GRAY_TEXT, labelpad=6
    )
    ax.tick_params(axis="x", colors=GRAY_TEXT, labelsize=10)
    ax.tick_params(axis="y", colors=GRAY_TEXT, labelsize=8.5)

    title = "BSW and MSW students recommend their placements at nearly identical rates"
    subtitle = "The difference between the two groups is small."
    _title(fig, title, subtitle)

    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.13, right=0.94)
    _despine(ax)
    _save(fig, "fig_16b_bsw_msw_recommendation.png")


# -----------------------------------------------------------------------------
# figure 17 - competency alignment scatter
# -----------------------------------------------------------------------------
def fig_17_competency_alignment(suf: pd.DataFrame) -> None:
    analysis_df = suf.dropna(
        subset=["mean_competency_score", "placement_quality_score"]
    )

    is_flag = analysis_df["concern_flag"].str.lower().eq("review recommended")
    is_mis = (
        analysis_df["misalignment_flag"]
        .str.lower()
        .eq("score-narrative mismatch")
    )

    background_df = analysis_df[~is_flag & ~is_mis]
    flagged_df = analysis_df[is_flag & ~is_mis]
    mismatch_df = analysis_df[is_mis]

    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)

    for spine_name, spine in ax.spines.items():
        if spine_name in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_color(GRAY_SPINE)
            spine.set_linewidth(0.8)

    ax.xaxis.grid(False)
    ax.yaxis.grid(False)

    ax.scatter(
        background_df["mean_competency_score"],
        background_df["placement_quality_score"],
        color="#D9E8F2",
        s=28,
        alpha=0.55,
        linewidths=0,
        zorder=2,
    )
    ax.scatter(
        flagged_df["mean_competency_score"],
        flagged_df["placement_quality_score"],
        color=MUTED_BLUE,
        s=32,
        alpha=0.75,
        linewidths=0,
        zorder=3,
    )

    ax.axhline(
        concern_fit_score,
        color=GRAY_REF,
        linewidth=0.9,
        linestyle=":",
        zorder=1,
    )
    x_min_label = analysis_df["mean_competency_score"].min() - 0.3
    ax.text(
        x_min_label,
        concern_fit_score + 0.03,
        f"Concern threshold  {concern_fit_score:.1f}",
        color=GRAY_REF,
        fontsize=7.5,
        va="bottom",
        ha="left",
    )

    ax.scatter(
        mismatch_df["mean_competency_score"],
        mismatch_df["placement_quality_score"],
        color=RED,
        s=110,
        alpha=0.95,
        edgecolors=WHITE,
        linewidths=0.8,
        marker="D",
        zorder=5,
    )

    for _, row in mismatch_df.iterrows():
        name = row["agency_name_display"]
        name = name[:38] + "…" if len(name) > 38 else name
        comp = row["mean_competency_score"]
        pqs = row["placement_quality_score"]
        sent = row["overall_sentiment_score"]

        detail = (
            f"Competency {comp:.1f}  |  Placement {pqs:.2f}"
            f"  |  Sentiment {sent:.3f}"
        )

        ax.annotate(
            name,
            xy=(comp, pqs),
            xytext=(0, 22),
            textcoords="offset points",
            fontsize=8.8,
            color=RED,
            fontweight="bold",
            ha="center",
            va="bottom",
            path_effects=[pe.withStroke(linewidth=2.5, foreground=WHITE)],
            zorder=6,
        )
        ax.annotate(
            detail,
            xy=(comp, pqs),
            xytext=(0, 10),
            textcoords="offset points",
            fontsize=7.8,
            color=RED,
            fontweight="bold",
            ha="center",
            va="bottom",
            path_effects=[pe.withStroke(linewidth=2.5, foreground=WHITE)],
            zorder=6,
        )

    x_pad = 0.4
    ax.set_xlim(
        analysis_df["mean_competency_score"].min() - x_pad,
        analysis_df["mean_competency_score"].max() + x_pad,
    )
    ax.set_ylim(1.8, 5.1)
    ax.set_xlabel(
        "Mean program competency score",
        fontsize=9,
        color=GRAY_TEXT,
        labelpad=6,
    )
    ax.set_ylabel(
        "Placement Quality Score", fontsize=9, color=GRAY_TEXT, labelpad=6
    )
    ax.tick_params(colors=GRAY_TEXT, labelsize=8.5)

    title = "Passing competency benchmarks does not predict a strong placement experience"
    subtitle = (
        "Competency scores and Placement Quality Scores show a weak correlation (r = -0.009, p = 0.89), "
        "indicating they are mostly unrelated.\nTwo agencies met the required standards, but students "
        "reported some of the poorest experiences in the dataset."
    )
    _title(fig, title, subtitle)

    fig.subplots_adjust(top=0.87, bottom=0.11, left=0.09, right=0.97)
    _despine(ax)
    _save(fig, "fig_17_competency_alignment_scatter.png")


# -----------------------------------------------------------------------------
# figure 17B - score range comparison
# -----------------------------------------------------------------------------
def fig_17b_score_range_comparison(suf: pd.DataFrame) -> None:
    comp_s = pd.to_numeric(
        suf["mean_competency_score"], errors="coerce"
    ).dropna()
    pqs_s = pd.to_numeric(
        suf["placement_quality_score"], errors="coerce"
    ).dropna()

    strips = [
        {
            "label": "Competency benchmark scores",
            "color": BLUE,
            "v_min": comp_s.min(),
            "v_mean": comp_s.mean(),
            "v_max": comp_s.max(),
            "x_lo": 84.5,
            "x_hi": 95.0,
            "x_ticks": [86, 88, 90, 92, 94],
            "fmt": ".1f",
            "span": f"Range spans {comp_s.max() - comp_s.min():.1f} points",
        },
        {
            "label": "Agency Placement Quality Score",
            "color": TEAL,
            "v_min": pqs_s.min(),
            "v_mean": pqs_s.mean(),
            "v_max": pqs_s.max(),
            "x_lo": 1.8,
            "x_hi": 5.4,
            "x_ticks": [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            "fmt": ".2f",
            "span": f"Range spans {pqs_s.max() - pqs_s.min():.2f} points",
        },
    ]

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.0))
    fig.patch.set_facecolor(WHITE)

    for ax, strip in zip(axes, strips):
        ax.set_facecolor(WHITE)

        for sp, spine in ax.spines.items():
            if sp == "bottom":
                spine.set_color(GRAY_SPINE)
                spine.set_linewidth(0.8)
            else:
                spine.set_visible(False)

        ax.set_xlim(strip["x_lo"], strip["x_hi"])
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.yaxis.grid(False)
        ax.xaxis.grid(False)
        ax.set_xticks(strip["x_ticks"])
        ax.set_xticklabels(
            [str(tick) for tick in strip["x_ticks"]],
            fontsize=8.5,
            color=GRAY_TEXT,
        )
        ax.tick_params(axis="x", length=4, color=GRAY_SPINE)

        color = strip["color"]
        y = 0.52

        ax.plot(
            [strip["v_min"], strip["v_max"]],
            [y, y],
            color=color,
            linewidth=4.0,
            solid_capstyle="round",
            zorder=3,
        )

        for value, stat_name in [
            (strip["v_min"], "Minimum"),
            (strip["v_max"], "Maximum"),
            (strip["v_mean"], "Mean"),
        ]:
            ax.scatter(
                value,
                y,
                s=110,
                color=color,
                edgecolors=WHITE,
                linewidths=1.8,
                zorder=5,
            )
            ax.text(
                value,
                y + 0.20,
                stat_name,
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=color,
                fontweight="bold",
            )
            ax.text(
                value,
                y - 0.20,
                f"{value:{strip['fmt']}}",
                ha="center",
                va="top",
                fontsize=10,
                color=color,
                fontweight="bold",
            )

        ax.text(
            strip["x_hi"],
            0.04,
            strip["span"],
            ha="right",
            va="bottom",
            fontsize=8,
            color=GRAY_TEXT,
            fontstyle="italic",
        )
        ax.text(
            strip["x_lo"],
            0.99,
            strip["label"],
            ha="left",
            va="top",
            fontsize=11,
            color=color,
            fontweight="bold",
        )

    title = "Competency benchmark scores stay tightly clustered, while agency Placement Quality Scores vary much more"
    subtitle = (
        f"Competency benchmark scores range from {comp_s.min():.1f} to {comp_s.max():.1f}, "
        f"while Placement Quality Scores range from {pqs_s.min():.1f} to {pqs_s.max():.1f}."
    )
    _title(fig, title, subtitle)

    fig.subplots_adjust(
        top=0.88, bottom=0.07, left=0.04, right=0.97, hspace=0.6
    )
    _save(fig, "fig_17b_score_range_comparison.png")


# -----------------------------------------------------------------------------
# figure 18 - pipeline diagram
# -----------------------------------------------------------------------------
def fig_18_pipeline_diagram() -> None:
    steps = [
        ("1", "Raw\nEvaluations", "One row per\nstudent response"),
        ("2", "Cleaning\n& Scoring", "Normalize, encode,\nscore sentiment"),
        ("3", "Agency\nProfiles", "Aggregate to\nagency level"),
        ("4", "Flags\n& Trends", "Concern and\nmisalignment logic"),
        ("5", "Figures\n& Dashboard", "Report and\nleadership view"),
    ]

    n_steps = len(steps)
    xs = [i / (n_steps - 1) for i in range(n_steps)]
    line_y = 0.78

    fig, ax = plt.subplots(figsize=(11.0, 3.8))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.05, 1.10)
    ax.axis("off")

    ax.plot(
        [xs[0], xs[-1]],
        [line_y, line_y],
        color=GRAY_SPINE,
        linewidth=1.5,
        zorder=1,
        solid_capstyle="round",
    )

    for i, (num, step_title, desc) in enumerate(steps):
        x = xs[i]

        ax.scatter(
            x,
            line_y,
            s=1400,
            color=TEAL,
            edgecolors=WHITE,
            linewidths=2.0,
            zorder=3,
        )
        ax.text(
            x,
            line_y,
            num,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=WHITE,
            zorder=4,
        )
        ax.text(
            x,
            line_y - 0.112,
            step_title,
            ha="center",
            va="top",
            fontsize=9.5,
            fontweight="bold",
            color=TEAL,
            linespacing=1.3,
            zorder=2,
        )
        ax.text(
            x,
            line_y - 0.362,
            desc,
            ha="center",
            va="top",
            fontsize=8,
            color=GRAY_TEXT,
            linespacing=1.40,
            zorder=2,
        )

    ax.text(
        -0.06,
        1.07,
        "End-to-end workflow: from raw evaluations to leadership review",
        ha="left",
        va="top",
        fontsize=11.5,
        fontweight="bold",
        color=TITLE,
    )

    _save(fig, "fig_18_pipeline_diagram.png")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def generate_all_figures() -> None:
    """Load pipeline outputs and build every report figure."""
    profiles_df = pd.read_csv(agency_profiles_file)
    trends_df = pd.read_csv(agency_trends_file)
    raw_df = pd.read_csv(input_file)
    row_df = pd.read_csv(evaluations_text_file)

    for col in [
        "overall_sentiment_score",
        "response_count",
        "concern_indicator_count",
        "mean_competency_score",
        "placement_quality_score",
        "recommendation_rate",
    ]:
        profiles_df[col] = pd.to_numeric(profiles_df[col], errors="coerce")

    profiles_df["concern_flag"] = profiles_df["concern_flag"].fillna("no flag")
    profiles_df["data_quality"] = profiles_df["data_quality"].fillna(
        "sufficient"
    )
    profiles_df["fit_trend"] = profiles_df["fit_trend"].fillna("stable")
    profiles_df["misalignment_flag"] = profiles_df["misalignment_flag"].fillna(
        "no flag"
    )

    suf = profiles_df[profiles_df["data_quality"] == "sufficient"].copy()

    fig_01_concern_flag_summary(suf)
    fig_02_fit_vs_recommendation(suf)
    fig_03_sentiment_distribution(suf)
    fig_04_likert_mean_scores(suf)
    fig_05_theme_frequency(suf)
    fig_06_bigrams_comparison(profiles_df)
    fig_07_wordcloud_most_helpful(row_df)
    fig_08_wordcloud_least_helpful(row_df)
    fig_09_top_flagged_agencies(suf)
    fig_10_yearly_fit_trend(trends_df)
    fig_11_yearly_evaluation_volume(trends_df)
    fig_12_fit_score_distribution(suf)
    fig_13_recommendation_rate_distribution(suf)
    fig_14_sentiment_by_trend(suf)
    fig_15_agency_trend_spotlight(trends_df, profiles_df)
    fig_16a_bsw_msw_dumbbell(raw_df)
    fig_16b_bsw_msw_recommendation(raw_df)
    fig_17_competency_alignment(suf)
    fig_17b_score_range_comparison(suf)
    fig_18_pipeline_diagram()


if __name__ == "__main__":
    generate_all_figures()
