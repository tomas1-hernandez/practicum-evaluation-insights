"""Main pipeline for practicum evaluation insights.

Run this from the project root:
    python pipeline.py
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from config import (
    agency_profiles_file,
    agency_trends_file,
    competency_file,
    concern_admin_overload,
    concern_fit_score,
    concern_recommendation,
    concern_sentiment,
    concern_supervision_least,
    concern_threshold,
    declining_threshold,
    evaluations_text_file,
    input_file,
    min_responses,
    misalignment_admin_ceiling,
    misalignment_comp_floor,
    misalignment_sentiment_ceiling,
    tables_dir,
)
from constants import (
    competency_cols,
    likert_cols,
    likert_map,
    placement_quality_cols,
    program_level_map,
    stopwords,
    text_cols,
)
from theme_lexicon import prompt_artifact_bigrams, theme_dictionary

nltk.download("vader_lexicon", quiet=True)

required_input_cols = {
    "academic_year",
    "agency_name",
    "program_year",
    "recommend",
    *likert_cols,
    *text_cols,
}


# -----------------------------------------------------------------------------
# text utilities
# -----------------------------------------------------------------------------
def clean_text(value: object) -> str | None:
    """Clean text into a standard lowercase format for later steps."""
    if not isinstance(value, str) or not value.strip():
        return None

    value = value.lower().strip()
    value = re.sub(r"[^a-z\s']", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value if len(value) > 2 else None


def normalize_agency_name(value: object) -> str:
    """Normalize agency names so small variants do not split the same site."""
    text = str(value).strip().lower().rstrip(".,;")
    return re.sub(r"\s+", " ", text)


def tokenize(value: str | None) -> list[str]:
    """Tokenize cleaned text and remove common filler words."""
    if not isinstance(value, str) or not value.strip():
        return []

    tokens = re.findall(r"\b[a-z]+\b", value)
    return [
        token for token in tokens if token not in stopwords and len(token) > 2
    ]


def get_bigrams(tokens: list[str]) -> list[str]:
    """Build bigrams and remove phrases that mostly repeat the survey prompt."""
    bigrams = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
    return [
        bigram for bigram in bigrams if bigram not in prompt_artifact_bigrams
    ]


def format_counts(counter_obj: Counter, top_n: int) -> str:
    """Format top counts into a simple pipe-separated string."""
    return " | ".join(
        f"{term}:{count}" for term, count in counter_obj.most_common(top_n)
    )


def score_sentiment_vader(
    analyzer: SentimentIntensityAnalyzer, value: object
) -> float | None:
    """Return the VADER compound score for one response."""
    if not isinstance(value, str) or not value.strip():
        return None
    return round(analyzer.polarity_scores(value)["compound"], 4)


# -----------------------------------------------------------------------------
# I/O utilities
# -----------------------------------------------------------------------------
def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a dataframe to csv and create the parent folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def validate_input_columns(df: pd.DataFrame) -> None:
    """Fail clearly if the raw file is missing required columns."""
    missing = sorted(required_input_cols.difference(df.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            "The raw evaluation file is missing required columns: "
            f"{missing_text}."
        )


# -----------------------------------------------------------------------------
# data structuring
# -----------------------------------------------------------------------------
def add_academic_year_start(df: pd.DataFrame) -> pd.DataFrame:
    """Pull the starting year from the academic year label."""
    df = df.copy()
    df["academic_year_start"] = (
        df["academic_year"]
        .astype("string")
        .str.extract(r"(\d{4})")
        .astype(float)
        .astype("Int64")
    )
    return df


def add_program_level(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw program_year values to bsw or msw."""
    df = df.copy()
    df["program_level"] = (
        df["program_year"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map(program_level_map)
    )
    return df


def get_agency_display_map(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the most common original agency name for display."""
    return (
        df.groupby("agency_name")["agency_name_raw"]
        .agg(
            lambda values: values.astype("string")
            .str.strip()
            .value_counts()
            .idxmax()
        )
        .reset_index()
        .rename(columns={"agency_name_raw": "agency_name_display"})
    )


# -----------------------------------------------------------------------------
# competency score join
# -----------------------------------------------------------------------------
def load_competency_scores() -> pd.DataFrame:
    """Load the program-level competency reference file."""
    comp_df = pd.read_csv(competency_file)
    comp_df["program_level"] = (
        comp_df["program_level"].astype("string").str.strip().str.lower()
    )
    comp_df["academic_year"] = (
        comp_df["academic_year"].astype("string").str.strip()
    )
    comp_df["mean_competency_score"] = (
        comp_df[competency_cols].mean(axis=1).round(2)
    )
    return comp_df


def join_competency_scores(
    evaluations_df: pd.DataFrame,
    comp_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join program-level competency scores onto evaluation rows."""
    comp_cols_to_join = [
        "program_level",
        "academic_year",
        "mean_competency_score",
    ] + competency_cols

    merged = evaluations_df.merge(
        comp_df[comp_cols_to_join],
        on=["program_level", "academic_year"],
        how="left",
    )

    matched = merged["mean_competency_score"].notna().sum()
    total = len(merged)
    print(
        f"  competency join: {matched}/{total} rows matched "
        f"({round(100 * matched / total, 1)}%)"
    )
    return merged


# -----------------------------------------------------------------------------
# NLP: topic modeling, word frequency, theme tagging
# -----------------------------------------------------------------------------
def build_lda_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create a topic summary from most_helpful and least_helpful text."""
    rows: list[dict[str, object]] = []

    for column in ["least_helpful_cleaned", "most_helpful_cleaned"]:
        documents = df[column].dropna().tolist()
        if len(documents) < 10:
            continue

        vectorizer = CountVectorizer(
            max_features=500,
            min_df=5,
            ngram_range=(1, 2),
            stop_words="english",
        )
        matrix = vectorizer.fit_transform(documents)

        lda = LatentDirichletAllocation(
            learning_method="batch",
            n_components=6,
            random_state=42,
        )
        lda.fit(matrix)
        words = vectorizer.get_feature_names_out()

        for topic_index, topic_weights in enumerate(lda.components_, start=1):
            top_word_indexes = topic_weights.argsort()[-10:][::-1]
            top_words = [words[index] for index in top_word_indexes]
            rows.append(
                {
                    "text_column": column.replace("_cleaned", ""),
                    "topic_index": topic_index,
                    "top_words": " | ".join(top_words),
                }
            )

    return pd.DataFrame(rows)


def build_word_freq_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize top words and bigrams by agency."""
    rows: list[dict[str, object]] = []

    for agency_name, agency_df in df.groupby("agency_name"):
        if len(agency_df) < min_responses:
            continue

        row = {
            "agency_name": agency_name,
            "agency_name_display": agency_df["agency_name_display"]
            .mode()
            .iloc[0],
            "response_count": len(agency_df),
        }

        for column in ["least_helpful", "most_helpful"]:
            token_counter: Counter = Counter()
            bigram_counter: Counter = Counter()

            for text in agency_df[f"{column}_cleaned"].dropna():
                tokens = tokenize(text)
                token_counter.update(tokens)
                bigram_counter.update(get_bigrams(tokens))

            row[f"{column}_top_words"] = format_counts(token_counter, 20)
            row[f"{column}_top_bigrams"] = format_counts(bigram_counter, 15)

        rows.append(row)

    return pd.DataFrame(rows)


def add_theme_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each response with theme hits from the project lexicon."""
    df = df.copy()

    for theme_name, keywords in theme_dictionary.items():
        df[f"{theme_name}_helpful"] = (
            df["most_helpful_cleaned"]
            .fillna("")
            .apply(
                lambda text: int(any(keyword in text for keyword in keywords))
            )
        )
        df[f"{theme_name}_least"] = (
            df["least_helpful_cleaned"]
            .fillna("")
            .apply(
                lambda text: int(any(keyword in text for keyword in keywords))
            )
        )

    return df


def build_agency_themes(df: pd.DataFrame) -> pd.DataFrame:
    """Roll response-level theme tags up to the agency level."""
    rows: list[dict[str, object]] = []

    for agency_name, agency_df in df.groupby("agency_name"):
        if len(agency_df) < min_responses:
            continue

        helpful_total = agency_df["most_helpful_cleaned"].notna().sum()
        least_total = agency_df["least_helpful_cleaned"].notna().sum()

        row = {
            "agency_name": agency_name,
            "agency_name_display": agency_df["agency_name_display"]
            .mode()
            .iloc[0],
            "response_count": len(agency_df),
        }

        for theme_name in sorted(theme_dictionary):
            helpful_matches = agency_df[f"{theme_name}_helpful"].sum()
            least_matches = agency_df[f"{theme_name}_least"].sum()

            row[f"{theme_name}_helpful_pct"] = (
                round(100 * helpful_matches / helpful_total, 1)
                if helpful_total
                else None
            )
            row[f"{theme_name}_least_pct"] = (
                round(100 * least_matches / least_total, 1)
                if least_total
                else None
            )

        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# scoring and aggregation
# -----------------------------------------------------------------------------
def build_grouped_scores(
    df: pd.DataFrame, group_cols: list[str]
) -> pd.DataFrame:
    """Build grouped structured-score summaries."""
    grouped = pd.concat(
        [
            df.groupby(group_cols).size().rename("response_count"),
            df.groupby(group_cols)[likert_cols].mean().round(2),
            df.groupby(group_cols)["recommend_num"]
            .mean()
            .round(2)
            .rename("recommendation_rate"),
        ],
        axis=1,
    ).reset_index()

    grouped["placement_quality_score"] = (
        grouped[placement_quality_cols].mean(axis=1).round(2)
    )
    grouped["data_quality"] = grouped["response_count"].apply(
        lambda count: "sufficient" if count >= min_responses else "limited"
    )
    return grouped


def build_grouped_sentiment(
    df: pd.DataFrame, group_cols: list[str]
) -> pd.DataFrame:
    """Build grouped sentiment summaries."""
    polarity_cols = [f"{column}_polarity" for column in text_cols]

    grouped = pd.concat(
        [
            df.groupby(group_cols).size().rename("response_count"),
            df.groupby(group_cols)[polarity_cols].mean().round(4),
        ],
        axis=1,
    ).reset_index()

    grouped["overall_sentiment_score"] = (
        grouped[polarity_cols].mean(axis=1).round(4)
    )
    grouped["data_quality"] = grouped["response_count"].apply(
        lambda count: "sufficient" if count >= min_responses else "limited"
    )
    return grouped


def build_agency_competency_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Average joined competency scores to the agency level."""
    rows: list[dict[str, object]] = []

    for agency_name, agency_df in df.groupby("agency_name"):
        if len(agency_df) < min_responses:
            continue

        row: dict[str, object] = {"agency_name": agency_name}
        row["mean_competency_score"] = round(
            agency_df["mean_competency_score"].dropna().mean(), 2
        )

        for col in competency_cols:
            row[f"agency_{col}_mean"] = round(
                agency_df[col].dropna().mean(), 2
            )

        rows.append(row)

    return pd.DataFrame(rows)


def build_response_completeness(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate average open-ended response length by agency."""
    df = df.copy()
    df["response_word_count"] = (
        df["most_helpful"]
        .fillna("")
        .apply(lambda text: len(str(text).split()))
    )
    return (
        df.groupby("agency_name")["response_word_count"]
        .mean()
        .round(1)
        .rename("mean_response_length")
        .reset_index()
    )


def build_recent_placement_scores(trends_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate a recent fit score from the two newest years per agency."""
    max_year = trends_df["academic_year_start"].max()
    recent_cutoff = max_year - 1
    recent_df = trends_df[
        trends_df["academic_year_start"] >= recent_cutoff
    ].copy()

    rows: list[dict[str, object]] = []
    for agency_name, agency_df in recent_df.groupby("agency_name"):
        total_responses = agency_df["response_count"].sum()
        weighted_fit = (
            agency_df["placement_quality_score"] * agency_df["response_count"]
        ).sum() / total_responses

        rows.append(
            {
                "agency_name": agency_name,
                "recent_placement_score": round(weighted_fit, 2),
            }
        )

    return pd.DataFrame(rows)


def add_placement_trend(profiles_df: pd.DataFrame) -> pd.DataFrame:
    """Label agencies as declining, improving, or stable."""
    df = profiles_df.copy()
    diff = df["recent_placement_score"] - df["placement_quality_score"]

    df["fit_trend"] = "stable"
    df.loc[diff <= -declining_threshold, "fit_trend"] = "declining"
    df.loc[diff >= declining_threshold, "fit_trend"] = "improving"
    return df


# -----------------------------------------------------------------------------
# concern and misalignment flags
# -----------------------------------------------------------------------------
def apply_concern_logic(
    df: pd.DataFrame,
    include_theme_signals: bool = True,
) -> pd.DataFrame:
    """Count concern signals and assign the final review flag."""
    df = df.copy()
    df["concern_indicator_count"] = 0

    df.loc[
        df["placement_quality_score"] < concern_fit_score,
        "concern_indicator_count",
    ] += 1
    df.loc[
        df["overall_sentiment_score"] < concern_sentiment,
        "concern_indicator_count",
    ] += 1
    df.loc[
        df["recommendation_rate"] < concern_recommendation,
        "concern_indicator_count",
    ] += 1

    if include_theme_signals:
        df.loc[
            df["administrative_overload_least_pct"] >= concern_admin_overload,
            "concern_indicator_count",
        ] += 1
        df.loc[
            df["strong_supervision_least_pct"] >= concern_supervision_least,
            "concern_indicator_count",
        ] += 1

    df["concern_flag"] = df["concern_indicator_count"].apply(
        lambda count: (
            "review recommended" if count >= concern_threshold else "no flag"
        )
    )
    return df


def apply_misalignment_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag agencies where benchmark scores look strong but the narrative does not."""
    df = df.copy()

    has_strong_competency = (
        df["mean_competency_score"] >= misalignment_comp_floor
    )
    has_low_sentiment = (
        df["overall_sentiment_score"] < misalignment_sentiment_ceiling
    )
    has_admin_overload = (
        df["administrative_overload_least_pct"] >= misalignment_admin_ceiling
    )

    df["misalignment_flag"] = "no flag"
    df.loc[
        has_strong_competency & has_low_sentiment & has_admin_overload,
        "misalignment_flag",
    ] = "score-narrative mismatch"

    return df


# -----------------------------------------------------------------------------
# yearly trends
# -----------------------------------------------------------------------------
def build_agency_yearly_trends(evaluations_df: pd.DataFrame) -> pd.DataFrame:
    """Build the agency-year trend table used in the report and app."""
    trend_scores = build_grouped_scores(
        evaluations_df,
        ["academic_year", "agency_name", "agency_name_display"],
    )
    trend_sentiment = build_grouped_sentiment(
        evaluations_df,
        ["academic_year", "agency_name", "agency_name_display"],
    )

    trends_df = trend_scores.merge(
        trend_sentiment.drop(columns=["response_count", "data_quality"]),
        on=["academic_year", "agency_name", "agency_name_display"],
        how="left",
    )

    trends_df = apply_concern_logic(trends_df, include_theme_signals=False)
    trends_df = add_academic_year_start(trends_df)

    priority_cols = [
        "academic_year",
        "academic_year_start",
        "agency_name",
        "agency_name_display",
        "response_count",
        "data_quality",
        "placement_quality_score",
        "recommendation_rate",
        "overall_sentiment_score",
        "concern_indicator_count",
        "concern_flag",
        "prepared_for_practice",
        "learning_goals_met",
        "supervision_frequency",
        "supervision_quality",
        "felt_prepared",
    ]
    remaining_cols = [
        col for col in trends_df.columns if col not in priority_cols
    ]

    return trends_df[priority_cols + remaining_cols].sort_values(
        ["academic_year_start", "agency_name_display"]
    )


# -----------------------------------------------------------------------------
# main pipeline
# -----------------------------------------------------------------------------
def run_pipeline() -> None:
    """Run the full pipeline from raw input to final outputs."""
    print("running practicum evaluation insights pipeline")

    evaluations_df = pd.read_csv(input_file)
    validate_input_columns(evaluations_df)
    evaluations_df = add_academic_year_start(evaluations_df)
    print(f"  loaded {len(evaluations_df)} rows from {input_file.name}")

    evaluations_df["agency_name_raw"] = (
        evaluations_df["agency_name"].astype("string").str.strip()
    )
    evaluations_df["agency_name"] = evaluations_df["agency_name_raw"].apply(
        normalize_agency_name
    )

    display_map_df = get_agency_display_map(evaluations_df)
    evaluations_df = evaluations_df.merge(
        display_map_df, on="agency_name", how="left"
    )
    evaluations_df = add_program_level(evaluations_df)

    for column in likert_cols:
        evaluations_df[column] = (
            evaluations_df[column]
            .astype("string")
            .str.strip()
            .str.lower()
            .map(likert_map)
        )

    evaluations_df["recommend_num"] = (
        evaluations_df["recommend"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map({"no": 0, "yes": 1})
    )

    vader_analyzer = SentimentIntensityAnalyzer()
    for column in text_cols:
        evaluations_df[f"{column}_cleaned"] = evaluations_df[column].apply(
            clean_text
        )
        evaluations_df[f"{column}_polarity"] = evaluations_df[column].apply(
            lambda value: score_sentiment_vader(vader_analyzer, value)
        )

    comp_df = load_competency_scores()
    evaluations_df = join_competency_scores(evaluations_df, comp_df)

    agency_scores_df = build_grouped_scores(
        evaluations_df,
        ["agency_name", "agency_name_display"],
    )
    save_csv(
        agency_scores_df.sort_values("agency_name_display"),
        tables_dir / "agency_scores.csv",
    )

    agency_sentiment_df = build_grouped_sentiment(
        evaluations_df,
        ["agency_name", "agency_name_display"],
    )
    save_csv(
        agency_sentiment_df.sort_values("agency_name_display"),
        tables_dir / "agency_sentiment.csv",
    )

    lda_topics_df = build_lda_summary(evaluations_df)
    save_csv(lda_topics_df, tables_dir / "lda_topics.csv")

    agency_word_freq_df = build_word_freq_summary(evaluations_df)
    save_csv(
        agency_word_freq_df.sort_values("agency_name_display"),
        tables_dir / "agency_word_freq.csv",
    )

    evaluations_df = add_theme_tags(evaluations_df)
    agency_themes_df = build_agency_themes(evaluations_df)
    save_csv(
        agency_themes_df.sort_values("agency_name_display"),
        tables_dir / "agency_themes.csv",
    )

    profiles_df = (
        agency_scores_df.merge(
            agency_sentiment_df.drop(
                columns=["response_count", "data_quality"]
            ),
            on=["agency_name", "agency_name_display"],
            how="left",
        )
        .merge(
            agency_word_freq_df.drop(columns=["response_count"]),
            on=["agency_name", "agency_name_display"],
            how="left",
        )
        .merge(
            agency_themes_df.drop(columns=["response_count"]),
            on=["agency_name", "agency_name_display"],
            how="left",
        )
    )

    agency_comp_df = build_agency_competency_summary(evaluations_df)
    profiles_df = profiles_df.merge(
        agency_comp_df, on="agency_name", how="left"
    )
    print(
        f"  competency summary joined for "
        f"{agency_comp_df['agency_name'].nunique()} agencies"
    )

    profiles_df = apply_concern_logic(profiles_df, include_theme_signals=True)
    profiles_df = apply_misalignment_flag(profiles_df)

    agency_yearly_trends_df = build_agency_yearly_trends(evaluations_df)
    save_csv(agency_yearly_trends_df, agency_trends_file)

    recent_fit_df = build_recent_placement_scores(agency_yearly_trends_df)
    profiles_df = profiles_df.merge(
        recent_fit_df, on="agency_name", how="left"
    )
    profiles_df = add_placement_trend(profiles_df)

    completeness_df = build_response_completeness(evaluations_df)
    profiles_df = profiles_df.merge(
        completeness_df, on="agency_name", how="left"
    )

    priority_cols = [
        "agency_name",
        "agency_name_display",
        "response_count",
        "data_quality",
        "placement_quality_score",
        "recent_placement_score",
        "fit_trend",
        "mean_response_length",
        "recommendation_rate",
        "overall_sentiment_score",
        "mean_competency_score",
        "concern_indicator_count",
        "concern_flag",
        "misalignment_flag",
        "most_helpful_polarity",
        "least_helpful_polarity",
        "strong_supervision_helpful_pct",
        "strong_supervision_least_pct",
        "direct_practice_opportunity_helpful_pct",
        "direct_practice_opportunity_least_pct",
        "administrative_overload_helpful_pct",
        "administrative_overload_least_pct",
        "organizational_structure_helpful_pct",
        "organizational_structure_least_pct",
        "learning_environment_helpful_pct",
        "learning_environment_least_pct",
        "social_justice_alignment_helpful_pct",
        "social_justice_alignment_least_pct",
        "most_helpful_top_words",
        "most_helpful_top_bigrams",
        "least_helpful_top_words",
        "least_helpful_top_bigrams",
    ]
    remaining_cols = [
        col for col in profiles_df.columns if col not in priority_cols
    ]
    ordered_cols = [
        col for col in priority_cols if col in profiles_df.columns
    ] + remaining_cols

    profiles_df = profiles_df[ordered_cols].sort_values(
        [
            "concern_indicator_count",
            "placement_quality_score",
            "agency_name_display",
        ],
        ascending=[False, True, True],
    )
    save_csv(profiles_df, agency_profiles_file)

    text_export_cols = [
        "academic_year",
        "agency_name",
        "agency_name_display",
    ] + text_cols
    save_csv(evaluations_df[text_export_cols], evaluations_text_file)

    flagged = int((profiles_df["concern_flag"] == "review recommended").sum())
    declining = int((profiles_df["fit_trend"] == "declining").sum())
    misaligned = int(
        (profiles_df["misalignment_flag"] == "score-narrative mismatch").sum()
    )
    included = int(profiles_df["most_helpful_top_words"].notna().sum())

    print(
        f"  {len(evaluations_df)} evaluations across "
        f"{profiles_df['agency_name'].nunique()} agencies "
        f"({included} with text-based profiles)"
    )
    print(f"  {flagged} agencies flagged for review")
    print(f"  {declining} agencies with a declining trend")
    print(f"  {misaligned} score-narrative mismatches")


if __name__ == "__main__":
    run_pipeline()
