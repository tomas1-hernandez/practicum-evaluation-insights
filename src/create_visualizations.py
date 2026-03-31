# create_visualizations.py
from __future__ import annotations

import os
import random
import re
from collections import Counter

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

from constants import likert_cols, stopwords

input_file = "data/raw/feedback_hernandez_20260326_v1.csv"
profiles_file = "outputs/agency_profiles/agency_profiles.csv"
trends_file = "outputs/tables/agency_yearly_trends.csv"
figures_dir = "outputs/figures"

color_positive = "#4f97b7"
color_concern = "#e6616b"
color_highlight = "#e7a66a"
color_neutral = "#888888"

likert_display = {
    "prepared_for_practice": "Prepared for practice",
    "learning_goals_met": "Learning goals met",
    "comp_1_professional_dev": "Comp 1: Professional dev",
    "comp_2_ethical_decisions": "Comp 2: Ethical decisions",
    "comp_3_critical_thinking": "Comp 3: Critical thinking",
    "comp_4_aradei": "Comp 4: ARADEI",
    "comp_5_human_rights_justice": "Comp 5: Human rights",
    "comp_6_research_practice": "Comp 6: Research practice",
    "comp_7_human_behavior": "Comp 7: Human behavior",
    "comp_8_policy_practice": "Comp 8: Policy practice",
    "comp_9_direct_practice": "Comp 9: Direct practice",
    "supervision_frequency": "Supervision frequency",
    "supervision_quality": "Supervision quality",
    "felt_prepared": "Felt prepared overall",
}

theme_labels = {
    "administrative_overload": "Administrative overload",
    "direct_practice_opportunity": "Direct practice opportunity",
    "learning_environment": "Learning environment",
    "organizational_structure": "Organizational structure",
    "social_justice_alignment": "Social justice alignment",
    "strong_supervision": "Strong supervision",
}


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "figure.dpi": 100,
    }
)


# helper function to save figures with consistent settings
def save_figure(fig, filename):
    os.makedirs(figures_dir, exist_ok=True)
    fig.savefig(
        os.path.join(figures_dir, filename),
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)


# helper function to build a word frequency dictionary from a text series,
# excluding stopwords and short words
def build_word_freq_dict(text_series: pd.Series, top_n: int = 150) -> dict[str, int]:
    tokens: list[str] = []
    for text in text_series.dropna():
        words = re.findall(r"\b[a-z]+\b", str(text).lower())
        tokens.extend([word for word in words if word not in stopwords and len(word) > 2])
    return dict(Counter(tokens).most_common(top_n))


def parse_bigram_string(value: str, top_n: int = 10) -> dict[str, int]:
    if not isinstance(value, str):
        return {}
    result = {}
    for item in value.split(" | ")[:top_n]:
        parts = item.rsplit(":", 1)
        if len(parts) == 2:
            try:
                result[parts[0].strip()] = int(parts[1].strip())
            except ValueError:
                pass
    return result


def blue_color_func(*args, **kwargs):
    return f"hsl(210, 80%, {random.randint(20, 50)}%)"


def red_color_func(*args, **kwargs):
    return f"hsl(5, 80%, {random.randint(20, 50)}%)"


def get_display_col(df: pd.DataFrame) -> str:
    return "agency_name_display" if "agency_name_display" in df.columns else "agency_name"


def generate_all_figures() -> None:
    profiles_df = pd.read_csv(profiles_file)
    sufficient_df = profiles_df[profiles_df["data_quality"] == "sufficient"].copy()
    row_df = pd.read_csv(input_file)
    trends_df = pd.read_csv(trends_file) if os.path.exists(trends_file) else None
    display_col = get_display_col(sufficient_df)

    # -- fig 01: concern flag bar chart --
    flagged_count = int((sufficient_df["concern_flag"] == "review recommended").sum())
    not_flagged_count = len(sufficient_df) - flagged_count
    flagged_share = round((flagged_count / len(sufficient_df)) * 100, 1) if len(sufficient_df) else 0

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(
        ["No flag", "Review recommended"],
        [not_flagged_count, flagged_count],
        color=[color_positive, color_concern],
        alpha=0.85,
    )
    for bar, count in zip(bars, [not_flagged_count, flagged_count]):
        ax.text(
            count + 1,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_xlabel("Number of agencies")
    ax.set_xlim(0, max(not_flagged_count, flagged_count) + 20)
    ax.set_title(
        f"Figure 1. {flagged_share}% of practicum agencies meet the\n" f"threshold for a leadership review",
        loc="left",
    )
    plt.tight_layout()
    save_figure(fig, "fig_01_concern_flag_summary.png")

    # -- fig 02: fit score vs recommendation rate --
    fig, ax = plt.subplots(figsize=(9, 6))
    mask = sufficient_df["concern_flag"] == "review recommended"
    recommendation_pct = sufficient_df["recommendation_rate"] * 100

    ax.scatter(
        sufficient_df.loc[~mask, "composite_fit_score"],
        recommendation_pct.loc[~mask],
        color=color_positive,
        alpha=0.6,
        s=50,
        label="No flag",
    )
    ax.scatter(
        sufficient_df.loc[mask, "composite_fit_score"],
        recommendation_pct.loc[mask],
        color=color_concern,
        alpha=0.7,
        s=60,
        marker="^",
        label="Flagged for review",
    )

    ax.axhline(70, color=color_neutral, linestyle="--", linewidth=1, alpha=0.6)
    ax.axvline(3.50, color=color_neutral, linestyle=":", linewidth=1, alpha=0.6)

    ax.set_xlabel("Composite fit score (1-5)")
    ax.set_ylabel("Recommendation rate (%)")
    ax.set_title(
        "Figure 2. Agencies with lower fit scores tend to get fewer recommendations,\n"
        "and are mostly found in the lower-left area of the chart.",
        loc="left",
    )
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    save_figure(fig, "fig_02_fit_vs_recommendation.png")

    # -- fig 03: sentiment distribution --
    flagged = sufficient_df[sufficient_df["concern_flag"] == "review recommended"]["overall_sentiment_score"]
    not_flagged = sufficient_df[sufficient_df["concern_flag"] == "no flag"]["overall_sentiment_score"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(not_flagged, bins=25, alpha=0.75, color=color_positive, label=f"No flag ({len(not_flagged)} agencies)")
    ax.hist(flagged, bins=25, alpha=0.75, color=color_concern, label=f"Flagged  ({len(flagged)} agencies)")
    ax.axvline(
        sufficient_df["overall_sentiment_score"].mean(),
        color=color_neutral,
        linestyle="--",
        linewidth=1.5,
        label=f"Mean  {sufficient_df['overall_sentiment_score'].mean():.3f}",
    )
    ax.set_xlabel("Overall sentiment score")
    ax.set_title(
        "Figure 3. Flagged agencies skew less positive in tone, though sentiment overlaps across both groups",
        loc="left",
    )
    ax.legend(frameon=False)
    ax.text(
        0.03,
        0.92,
        "The flagged group shifts left,\nbut the two distributions still overlap.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#121212",
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8),
    )
    plt.tight_layout()
    save_figure(fig, "fig_03_sentiment_distribution.png")

    # -- fig 04: mean likert scores --
    mean_scores = sufficient_df[likert_cols].mean().round(2)
    colors = [
        color_concern if score < 3.7 else color_highlight if score < 3.9 else color_positive for score in mean_scores
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh([likert_display[col] for col in likert_cols], mean_scores, color=colors, alpha=0.85)
    ax.axvline(mean_scores.mean(), color="#121212", linestyle=":", linewidth=1.5)
    ax.set_xlim(1, 5)
    ax.set_xlabel("Mean score  (1 = strongly disagree,  5 = strongly agree)")
    ax.set_title(
        "Figure 4. Students give high ratings to supervision,\n"
        "but they find preparedness and direct practice to be less satisfactory",
        loc="left",
    )
    for bar, score in zip(bars, mean_scores):
        ax.text(score + 0.05, bar.get_y() + bar.get_height() / 2, f"{score:.2f}", va="center", ha="left", fontsize=8.5)
    ax.legend(
        handles=[
            mpatches.Patch(color=color_concern, label="Below 3.70"),
            mpatches.Patch(color=color_highlight, label="3.70 – 3.89"),
            mpatches.Patch(color=color_positive, label="3.90 and above"),
        ],
        frameon=False,
        fontsize=8,
        loc="lower right",
    )
    plt.tight_layout()
    save_figure(fig, "fig_04_likert_mean_scores.png")

    # -- fig 05: theme frequency --
    theme_names = list(theme_labels)
    labels = [theme_labels[name] for name in theme_names]
    helpful = [sufficient_df[f"{name}_helpful_pct"].mean() for name in theme_names]
    least = [sufficient_df[f"{name}_least_pct"].mean() for name in theme_names]

    fig, ax = plt.subplots(figsize=(11, 7))
    x = range(len(labels))
    width = 0.38
    ax.bar(
        [value - width / 2 for value in x],
        helpful,
        width=width,
        color=color_positive,
        alpha=0.85,
        label="Appeared in 'most helpful'",
    )
    ax.bar(
        [value + width / 2 for value in x],
        least,
        width=width,
        color=color_concern,
        alpha=0.85,
        label="Appeared in 'least helpful'",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Mean % of responses tagged")
    ax.set_title(
        "Figure 5. Administrative overload appears far more often in least-helpful comments than in most-helpful ones",
        loc="left",
    )
    ax.legend(frameon=False)
    plt.tight_layout()
    save_figure(fig, "fig_05_theme_frequency.png")

    # -- fig 06: bigram comparison --
    helpful_bigrams = Counter()
    least_bigrams = Counter()
    for value in profiles_df["most_helpful_top_bigrams"].dropna():
        helpful_bigrams.update(parse_bigram_string(value, 15))
    for value in profiles_df["least_helpful_top_bigrams"].dropna():
        least_bigrams.update(parse_bigram_string(value, 15))

    top_helpful = helpful_bigrams.most_common(12)
    top_least = least_bigrams.most_common(12)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))
    ax_left.barh(
        [item[0] for item in reversed(top_helpful)],
        [item[1] for item in reversed(top_helpful)],
        color=color_positive,
        alpha=0.85,
    )
    ax_left.set_title("Most helpful — top phrases", loc="left", fontweight="bold")
    ax_left.tick_params(axis="y", labelsize=9)

    ax_right.barh(
        [item[0] for item in reversed(top_least)],
        [item[1] for item in reversed(top_least)],
        color=color_concern,
        alpha=0.85,
    )
    ax_right.set_title("Least helpful — top phrases", loc="left", fontweight="bold")
    ax_right.tick_params(axis="y", labelsize=9)

    fig.suptitle(
        "Figure 6. Helpful comments emphasize confidence and practice, while least helpful comments emphasize admin work and limited contact",
        fontsize=11,
        fontweight="bold",
        x=0.02,
        y=1.01,
        ha="left",
    )
    plt.tight_layout()
    save_figure(fig, "fig_06_bigrams_comparison.png")

    # -- fig 07: word cloud most helpful --
    helpful_freq = build_word_freq_dict(row_df["most_helpful"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(
        WordCloud(
            width=1000, height=500, background_color="white", color_func=blue_color_func, max_words=80
        ).generate_from_frequencies(helpful_freq),
        interpolation="bilinear",
    )
    ax.axis("off")
    ax.set_title(
        "Figure 7. Students most often describe support, learning, and practice as the\n"
        "most helpful parts of placement",
        pad=12,
        loc="left",
    )
    save_figure(fig, "fig_07_wordcloud_most_helpful.png")

    # -- fig 08: word cloud least helpful --
    least_freq = build_word_freq_dict(row_df["least_helpful"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(
        WordCloud(
            width=1000, height=500, background_color="white", color_func=red_color_func, max_words=80
        ).generate_from_frequencies(least_freq),
        interpolation="bilinear",
    )
    ax.axis("off")
    ax.set_title(
        "Figure 8. Student frustrations most often mention administrative work,\n"
        "limited practice, and not enough contact",
        pad=12,
        loc="left",
    )
    save_figure(fig, "fig_08_wordcloud_least_helpful.png")

    # -- fig 09: top 10 flagged agencies --
    flagged_df = (
        sufficient_df[sufficient_df["concern_flag"] == "review recommended"]
        .sort_values("composite_fit_score")
        .head(10)
        .copy()
    )

    flagged_df["label"] = flagged_df[display_col] + " (n=" + flagged_df["response_count"].astype(str) + ")"

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        flagged_df["label"],
        flagged_df["composite_fit_score"],
        color=color_concern,
        alpha=0.8,
    )
    ax.axvline(
        3.5,
        color=color_neutral,
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label="Current low-fit cut point 3.5",
    )
    ax.set_xlim(0, 5)
    ax.set_xlabel("Composite fit score (1-5)")
    ax.set_title(
        "Figure 9. Among agencies flagged for review, these ten have\n" "the lowest composite fit scores",
        loc="left",
    )
    ax.tick_params(axis="y", labelsize=8.5)

    for bar, score in zip(bars, flagged_df["composite_fit_score"]):
        ax.text(
            score + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.2f}",
            va="center",
            ha="left",
            fontsize=8,
        )

    ax.legend(frameon=False)
    plt.tight_layout()
    save_figure(fig, "fig_09_top_flagged_agencies.png")

    # -- fig 10: yearly trends --
    if trends_df is not None and not trends_df.empty:
        trends_df = trends_df.sort_values("academic_year_start")

        yearly = (
            trends_df.groupby("academic_year")
            .apply(
                lambda group: pd.Series(
                    {
                        "academic_year_start": group["academic_year_start"].iloc[0],
                        "evaluations": int(group["response_count"].sum()),
                        "mean_fit_score": round(
                            (group["composite_fit_score"] * group["response_count"]).sum()
                            / group["response_count"].sum(),
                            2,
                        ),
                    }
                ),
                include_groups=False,
            )
            .reset_index()
            .sort_values("academic_year_start")
        )

        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(11, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 1.15]},
        )

        # top chart: mean fit score
        ax1.plot(
            yearly["academic_year"],
            yearly["mean_fit_score"],
            color=color_positive,
            linewidth=2.5,
            label="Mean fit score",
        )
        ax1.set_ylabel("Mean fit score")
        ax1.set_ylim(3.2, 4.6)
        ax1.grid(axis="y", alpha=0.2)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        for year, score in zip(yearly["academic_year"], yearly["mean_fit_score"]):
            ax1.text(
                year,
                score + 0.04,
                f"{score:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax1.axhline(3.5, color=color_neutral, linestyle=":", linewidth=1, alpha=0.7)

        # bottom chart: evaluation volume
        bars = ax2.bar(
            yearly["academic_year"],
            yearly["evaluations"],
            color="#aad2e4ff",
            alpha=0.95,
            label="Evaluations",
        )
        ax2.set_ylabel("Number of evaluations")
        ax2.set_xlabel("Academic year")
        ax2.grid(axis="y", alpha=0.2)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.tick_params(axis="x", rotation=45)

        for bar, value in zip(bars, yearly["evaluations"]):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 4,
                f"{value}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        fig.suptitle(
            "Figure 10. Evaluation volume increased in recent years, while average fit scores were lower in the newest cohorts",
            x=0.01,
            ha="left",
            fontsize=12,
            fontweight="bold",
        )

        plt.tight_layout()
        save_figure(fig, "fig_10_yearly_fit_trend.png")


if __name__ == "__main__":
    generate_all_figures()
