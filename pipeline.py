"""Main pipeline for practicum evaluation intelligence.

This script takes the raw practicum evaluation export and turns it into the
agency-level files used in the report and dashboard. In plain terms, it cleans
messy multi-year data, scores the structured questions, tags the open-ended
comments, builds yearly trends, joins program-level competency scores, and
writes the final tables leadership can review.

The output is not meant to replace field judgment. It is meant to give field
education leaders a more organized place to start.

Run this from the project root:
    python pipeline.py

All outputs are written to outputs/agency_profiles/ and outputs/tables/.
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
    agency_trends_file,
    competency_file,
    concern_admin_overload,
    concern_fit_score,
    concern_recommendation,
    concern_sentiment,
    concern_supervision_least,
    concern_threshold,
    declining_threshold,
    input_file,
    min_responses,
    misalignment_admin_ceiling,
    misalignment_comp_floor,
    misalignment_sentiment_ceiling,
    profiles_dir,
    tables_dir,
)
from constants import fit_score_cols, likert_cols, stopwords, text_cols
from theme_lexicon import prompt_artifact_bigrams, theme_dictionary

nltk.download("vader_lexicon", quiet=True)

likert_map = {
    "neither agree nor disagree": 3,
    "somewhat agree": 4,
    "somewhat disagree": 2,
    "strongly agree": 5,
    "strongly disagree": 1,
}

# maps the raw program_year values in the evaluation data to the two program
# levels used in the competency scores file (bsw or msw)
# generalist and specialization are both msw-level placements at umssw
program_level_map = {
    "bsw": "bsw",
    "generalist": "msw",
    "specialization": "msw",
}

# competency column names as they appear in competency_scores_by_year.csv
competency_cols = [
    "competency_1",
    "competency_2",
    "competency_3",
    "competency_4",
    "competency_5",
    "competency_6",
    "competency_7",
    "competency_8",
    "competency_9",
]

required_input_cols = {
    "academic_year",
    "agency_name",
    "program_year",
    "recommend",
    *likert_cols,
    *text_cols,
}


# ------------------------------------------------------------------------------
# Text utilities
# ------------------------------------------------------------------------------


def clean_text(value: object) -> str | None:
    """Clean a text value so later text steps are working from the same format."""
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.lower().strip()
    value = re.sub(r"[^a-z\s']", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value if len(value) > 2 else None


def normalize_agency_name(value: object) -> str:
    """Normalize agency names so the same site is not split across tiny variants."""
    text = str(value).strip().lower().rstrip(".,;")
    return re.sub(r"\s+", " ", text)


def tokenize(value: str | None) -> list[str]:
    """Tokenize cleaned text and drop common filler words."""
    if not isinstance(value, str) or not value.strip():
        return []
    tokens = re.findall(r"\b[a-z]+\b", value)
    return [token for token in tokens if token not in stopwords and len(token) > 2]


def get_bigrams(tokens: list[str]) -> list[str]:
    """Build bigrams and remove phrases that mostly echo the survey prompt."""
    bigrams = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
    return [bigram for bigram in bigrams if bigram not in prompt_artifact_bigrams]


def format_counts(counter_obj: Counter, top_n: int) -> str:
    """Format a counter into a simple pipe-separated summary string."""
    return " | ".join(f"{term}:{count}" for term, count in counter_obj.most_common(top_n))


def score_sentiment_vader(analyzer: SentimentIntensityAnalyzer, value: object) -> float | None:
    """Return the VADER compound score for one response, or None if it is blank."""
    if not isinstance(value, str) or not value.strip():
        return None
    return round(analyzer.polarity_scores(value)["compound"], 4)


# ------------------------------------------------------------------------------
# I/O utilities
# ------------------------------------------------------------------------------


def save_csv(df: pd.DataFrame, folder: Path, filename: str) -> None:
    """Write a dataframe to csv and create the folder first if needed."""
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / filename, index=False)


def validate_input_columns(df: pd.DataFrame) -> None:
    """Fail clearly if the raw file is missing columns the pipeline depends on."""
    missing = sorted(required_input_cols.difference(df.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            "The raw evaluation file is missing required columns: "
            f"{missing_text}. Update the input file or the column mapping before rerunning."
        )


# ------------------------------------------------------------------------------
# Data structuring
# ------------------------------------------------------------------------------


def add_academic_year_start(df: pd.DataFrame) -> pd.DataFrame:
    """Pull the starting year out of the academic year label for sorting and trends."""
    df = df.copy()
    df["academic_year_start"] = (
        df["academic_year"].astype("string").str.extract(r"(\d{4})").astype(float).astype("Int64")
    )
    return df


def add_program_level(df: pd.DataFrame) -> pd.DataFrame:
    """Map the raw program_year field to a two-level program_level column.

    The raw data uses three values: bsw, generalist, and specialization.
    Generalist and specialization are both msw-level placements, so they
    are combined into a single msw label here. This is the key used to
    join competency scores from the separate reference file.
    """
    df = df.copy()
    df["program_level"] = (
        df["program_year"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map(program_level_map)
    )
    unmapped = df["program_level"].isna().sum()
    if unmapped > 0:
        print(f"  warning: {unmapped} rows have unrecognized program_year values and will not join competency scores")
    return df


def get_agency_display_map(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the most common original agency name for display in tables and charts."""
    return (
        df.groupby("agency_name")["agency_name_raw"]
        .agg(lambda values: values.astype("string").str.strip().value_counts().idxmax())
        .reset_index()
        .rename(columns={"agency_name_raw": "agency_name_display"})
    )


# ------------------------------------------------------------------------------
# Competency score join
# ------------------------------------------------------------------------------


def load_competency_scores() -> pd.DataFrame | None:
    """Load the program-level competency reference file if it exists.

    Returns None if the file is not present so the pipeline can still run
    without it - the misalignment flag will simply not be calculated.
    """
    if not competency_file.exists():
        print(f"  note: competency file not found at {competency_file} — skipping competency join")
        return None

    comp_df = pd.read_csv(competency_file)
    comp_df["program_level"] = comp_df["program_level"].str.strip().str.lower()
    comp_df["academic_year"] = comp_df["academic_year"].astype("string").str.strip()

    # add a single mean across all nine competencies for use in the misalignment flag
    comp_df["mean_competency_score"] = comp_df[competency_cols].mean(axis=1).round(2)
    return comp_df


def join_competency_scores(evaluations_df: pd.DataFrame, comp_df: pd.DataFrame | None) -> pd.DataFrame:
    """Join program-level competency scores onto the evaluation rows.

    Each evaluation row gets the competency scores for its program level
    and academic year. This is a reference join - the competency scores
    represent program-wide benchmarks, not individual student scores.
    The join key is program_level + academic_year.

    Rows that do not match (unmapped program_year values, or years not in
    the competency file) will have null competency columns and will not
    contribute to the misalignment flag.
    """
    if comp_df is None:
        return evaluations_df

    comp_cols_to_join = ["program_level", "academic_year", "mean_competency_score"] + competency_cols
    merged = evaluations_df.merge(
        comp_df[comp_cols_to_join],
        on=["program_level", "academic_year"],
        how="left",
    )

    matched = merged["mean_competency_score"].notna().sum()
    total = len(merged)
    print(f"  competency join: {matched}/{total} rows matched ({round(100 * matched / total, 1)}%)")
    return merged


# ------------------------------------------------------------------------------
# NLP: topic modeling, word frequency, theme tagging
# ------------------------------------------------------------------------------


def build_lda_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create a topic summary so the repo keeps one broader text view too."""
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
    """Summarize top words and bigrams by agency so the outputs stay readable."""
    rows: list[dict[str, object]] = []

    for agency_name, agency_df in df.groupby("agency_name"):
        if len(agency_df) < min_responses:
            continue

        row = {
            "agency_name": agency_name,
            "agency_name_display": agency_df["agency_name_display"].mode().iloc[0],
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
    """Tag each response with simple theme hits using the project lexicon."""
    df = df.copy()
    for theme_name, keywords in theme_dictionary.items():
        df[f"{theme_name}_helpful"] = df["most_helpful_cleaned"].fillna("").apply(
            lambda text: int(any(keyword in text for keyword in keywords))
        )
        df[f"{theme_name}_least"] = df["least_helpful_cleaned"].fillna("").apply(
            lambda text: int(any(keyword in text for keyword in keywords))
        )
    return df


def build_agency_themes(df: pd.DataFrame) -> pd.DataFrame:
    """Roll theme tags up to the agency level for the dashboard and report."""
    rows: list[dict[str, object]] = []

    for agency_name, agency_df in df.groupby("agency_name"):
        if len(agency_df) < min_responses:
            continue

        helpful_total = agency_df["most_helpful_cleaned"].notna().sum()
        least_total = agency_df["least_helpful_cleaned"].notna().sum()

        row = {
            "agency_name": agency_name,
            "agency_name_display": agency_df["agency_name_display"].mode().iloc[0],
            "response_count": len(agency_df),
        }

        for theme_name in sorted(theme_dictionary):
            helpful_matches = agency_df[f"{theme_name}_helpful"].sum()
            least_matches = agency_df[f"{theme_name}_least"].sum()
            row[f"{theme_name}_helpful_pct"] = (
                round(100 * helpful_matches / helpful_total, 1) if helpful_total else None
            )
            row[f"{theme_name}_least_pct"] = (
                round(100 * least_matches / least_total, 1) if least_total else None
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
# Scoring and aggregation
# ------------------------------------------------------------------------------


def build_grouped_scores(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Build agency or agency-year score summaries from the structured columns."""
    grouped = pd.concat(
        [
            df.groupby(group_cols).size().rename("response_count"),
            df.groupby(group_cols)[likert_cols].mean().round(2),
            df.groupby(group_cols)["recommend_num"].mean().round(2).rename("recommendation_rate"),
        ],
        axis=1,
    ).reset_index()

    grouped["composite_fit_score"] = grouped[fit_score_cols].mean(axis=1).round(2)
    grouped["data_quality"] = grouped["response_count"].apply(
        lambda count: "sufficient" if count >= min_responses else "limited"
    )
    return grouped


def build_grouped_sentiment(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Build agency or agency-year sentiment summaries from the text columns."""
    polarity_cols = [f"{column}_polarity" for column in text_cols]
    grouped = pd.concat(
        [
            df.groupby(group_cols).size().rename("response_count"),
            df.groupby(group_cols)[polarity_cols].mean().round(4),
        ],
        axis=1,
    ).reset_index()

    grouped["overall_sentiment_score"] = grouped[polarity_cols].mean(axis=1).round(4)
    grouped["data_quality"] = grouped["response_count"].apply(
        lambda count: "sufficient" if count >= min_responses else "limited"
    )
    return grouped


def build_agency_competency_summary(df: pd.DataFrame) -> pd.DataFrame | None:
    """Average the joined competency scores to the agency level.

    Because competency scores are program-level benchmarks (not per-student),
    we take the mean across the academic years that each agency appears in,
    weighted equally. This gives each agency a single representative
    competency context for the misalignment analysis.

    Returns None if no competency columns are present in the dataframe.
    """
    if "mean_competency_score" not in df.columns:
        return None

    rows: list[dict[str, object]] = []
    for agency_name, agency_df in df.groupby("agency_name"):
        if len(agency_df) < min_responses:
            continue

        row: dict[str, object] = {"agency_name": agency_name}

        # mean competency score across the years this agency has data
        row["mean_competency_score"] = round(
            agency_df["mean_competency_score"].dropna().mean(), 2
        )

        # individual competency means - useful for drill-down in dashboard
        for col in competency_cols:
            if col in agency_df.columns:
                row[f"agency_{col}_mean"] = round(agency_df[col].dropna().mean(), 2)

        rows.append(row)

    return pd.DataFrame(rows) if rows else None


def build_response_completeness(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate how full the open-ended feedback is at each agency."""
    df = df.copy()
    df["response_word_count"] = df["most_helpful"].fillna("").apply(
        lambda text: len(str(text).split())
    )
    return (
        df.groupby("agency_name")["response_word_count"]
        .mean()
        .round(1)
        .rename("mean_response_length")
        .reset_index()
    )


def build_recent_fit_scores(trends_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate a recent fit score based on the two newest years per agency."""
    max_year = trends_df["academic_year_start"].max()
    recent_cutoff = max_year - 1
    recent_df = trends_df[trends_df["academic_year_start"] >= recent_cutoff].copy()

    rows: list[dict[str, object]] = []
    for agency_name, agency_df in recent_df.groupby("agency_name"):
        total_responses = agency_df["response_count"].sum()
        if total_responses == 0:
            continue
        weighted_fit = (
            agency_df["composite_fit_score"] * agency_df["response_count"]
        ).sum() / total_responses
        rows.append({"agency_name": agency_name, "recent_fit_score": round(weighted_fit, 2)})

    return pd.DataFrame(rows)


def add_fit_trend(profiles_df: pd.DataFrame) -> pd.DataFrame:
    """Label agencies as declining, improving, or stable based on recent fit change."""
    df = profiles_df.copy()
    if "recent_fit_score" not in df.columns or "composite_fit_score" not in df.columns:
        return df

    diff = df["recent_fit_score"] - df["composite_fit_score"]
    df["fit_trend"] = "stable"
    df.loc[diff <= -declining_threshold, "fit_trend"] = "declining"
    df.loc[diff >= declining_threshold, "fit_trend"] = "improving"
    return df


# ------------------------------------------------------------------------------
# Concern and misalignment flagging
# ------------------------------------------------------------------------------


def apply_concern_logic(df: pd.DataFrame) -> pd.DataFrame:
    """Count concern signals and assign the final review flag.

    The point is to avoid treating one weak metric like a final verdict.
    Agencies are only flagged when more than one concern signal lines up at
    the same time. Each indicator adds one point to the count.
    """
    df = df.copy()
    df["concern_indicator_count"] = 0

    df.loc[df["composite_fit_score"] < concern_fit_score, "concern_indicator_count"] += 1
    df.loc[df["overall_sentiment_score"] < concern_sentiment, "concern_indicator_count"] += 1

    if "administrative_overload_least_pct" in df.columns:
        df.loc[
            df["administrative_overload_least_pct"] >= concern_admin_overload,
            "concern_indicator_count",
        ] += 1

    if "strong_supervision_least_pct" in df.columns:
        df.loc[
            df["strong_supervision_least_pct"] >= concern_supervision_least,
            "concern_indicator_count",
        ] += 1

    df.loc[df["recommendation_rate"] < concern_recommendation, "concern_indicator_count"] += 1

    df["concern_flag"] = df["concern_indicator_count"].apply(
        lambda count: "review recommended" if count >= concern_threshold else "no flag"
    )
    return df


def apply_misalignment_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag agencies where program competency scores look strong but the student
    narrative tells a different story.

    This is a separate signal from the main concern flag. It is designed to
    surface agencies where students are meeting the formal competency benchmarks
    on paper, but their open-ended descriptions suggest the placement experience
    was limited, administrative-heavy, or unsupportive.

    The flag fires when all three conditions are true:
        1. The agency's mean program competency score is above the floor
           (students in that program/year cohort are passing benchmarks)
        2. The agency's overall sentiment score is below the ceiling
           (the language in open-ended comments skews negative or flat)
        3. Administrative overload appears in more than the ceiling pct
           of least-helpful responses (students describe admin-heavy work)

    Requiring all three conditions prevents the flag from firing on agencies
    that simply have one weak metric. The misalignment is meaningful only when
    the numbers say one thing and the words say another simultaneously.
    """
    df = df.copy()

    # default to no flag if competency data is not available
    if "mean_competency_score" not in df.columns:
        df["misalignment_flag"] = "no data"
        return df

    has_strong_comps = df["mean_competency_score"] >= misalignment_comp_floor
    has_low_sentiment = df["overall_sentiment_score"] < misalignment_sentiment_ceiling
    has_admin_overload = (
        df["administrative_overload_least_pct"] >= misalignment_admin_ceiling
        if "administrative_overload_least_pct" in df.columns
        else pd.Series(False, index=df.index)
    )

    df["misalignment_flag"] = "no flag"
    df.loc[
        has_strong_comps & has_low_sentiment & has_admin_overload,
        "misalignment_flag",
    ] = "score-narrative mismatch"

    return df


# ------------------------------------------------------------------------------
# Yearly trends
# ------------------------------------------------------------------------------


def build_agency_yearly_trends(evaluations_df: pd.DataFrame) -> pd.DataFrame:
    """Build the dedicated agency-year trend table used in the report and app."""
    trend_scores = build_grouped_scores(
        evaluations_df, ["academic_year", "agency_name", "agency_name_display"]
    )
    trend_sentiment = build_grouped_sentiment(
        evaluations_df, ["academic_year", "agency_name", "agency_name_display"]
    )

    trends_df = trend_scores.merge(
        trend_sentiment.drop(columns=["response_count", "data_quality"], errors="ignore"),
        on=["academic_year", "agency_name", "agency_name_display"],
        how="left",
    )

    trends_df = apply_concern_logic(trends_df)
    trends_df = add_academic_year_start(trends_df)

    priority_cols = [
        "academic_year",
        "academic_year_start",
        "agency_name",
        "agency_name_display",
        "response_count",
        "data_quality",
        "composite_fit_score",
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
    remaining_cols = [col for col in trends_df.columns if col not in priority_cols]
    return trends_df[priority_cols + remaining_cols].sort_values(
        ["academic_year_start", "agency_name_display"]
    )


# ------------------------------------------------------------------------------
# Pipeline entry point
# ------------------------------------------------------------------------------


def run_pipeline() -> None:
    """Run the full capstone pipeline from raw file to final agency outputs."""
    print("running practicum evaluation intelligence pipeline")

    # ----------------------------------------------------------------
    # phase 1: load and validate
    # ----------------------------------------------------------------
    evaluations_df = pd.read_csv(input_file)
    validate_input_columns(evaluations_df)
    evaluations_df = add_academic_year_start(evaluations_df)
    print(f"  loaded {len(evaluations_df)} rows from {input_file.name}")

    # ----------------------------------------------------------------
    # phase 2: standardize names and encode structured columns
    # ----------------------------------------------------------------
    evaluations_df["agency_name_raw"] = evaluations_df["agency_name"].astype("string").str.strip()
    evaluations_df["agency_name"] = evaluations_df["agency_name_raw"].apply(normalize_agency_name)

    display_map_df = get_agency_display_map(evaluations_df)
    evaluations_df = evaluations_df.merge(display_map_df, on="agency_name", how="left")

    # map program_year to program_level for the competency join
    evaluations_df = add_program_level(evaluations_df)

    for column in likert_cols:
        evaluations_df[column] = (
            evaluations_df[column].astype("string").str.strip().str.lower().map(likert_map)
        )

    evaluations_df["recommend_num"] = (
        evaluations_df["recommend"].astype("string").str.strip().str.lower().map({"no": 0, "yes": 1})
    )

    # ----------------------------------------------------------------
    # phase 3: sentiment scoring and text cleaning
    # ----------------------------------------------------------------
    vader_analyzer = SentimentIntensityAnalyzer()
    for column in text_cols:
        evaluations_df[f"{column}_cleaned"] = evaluations_df[column].apply(clean_text)
        evaluations_df[f"{column}_polarity"] = evaluations_df[column].apply(
            lambda value: score_sentiment_vader(vader_analyzer, value)
        )

    # ----------------------------------------------------------------
    # phase 4: join program-level competency scores
    # ----------------------------------------------------------------
    comp_df = load_competency_scores()
    evaluations_df = join_competency_scores(evaluations_df, comp_df)

    # ----------------------------------------------------------------
    # phase 5: build agency-level structured outputs
    # ----------------------------------------------------------------
    agency_scores_df = build_grouped_scores(evaluations_df, ["agency_name", "agency_name_display"])
    save_csv(agency_scores_df.sort_values("agency_name_display"), tables_dir, "agency_scores.csv")

    agency_sentiment_df = build_grouped_sentiment(
        evaluations_df, ["agency_name", "agency_name_display"]
    )
    save_csv(
        agency_sentiment_df.sort_values("agency_name_display"), tables_dir, "agency_sentiment.csv"
    )

    # ----------------------------------------------------------------
    # phase 6: NLP - topic modeling, word frequency, theme tagging
    # ----------------------------------------------------------------
    lda_topics_df = build_lda_summary(evaluations_df)
    save_csv(lda_topics_df, tables_dir, "lda_topics.csv")

    agency_word_freq_df = build_word_freq_summary(evaluations_df)
    save_csv(
        agency_word_freq_df.sort_values("agency_name_display"), tables_dir, "agency_word_freq.csv"
    )

    evaluations_df = add_theme_tags(evaluations_df)
    agency_themes_df = build_agency_themes(evaluations_df)
    save_csv(
        agency_themes_df.sort_values("agency_name_display"), tables_dir, "agency_themes.csv"
    )

    # ----------------------------------------------------------------
    # phase 7: merge into agency profiles and apply flags
    # ----------------------------------------------------------------
    profiles_df = (
        agency_scores_df.merge(
            agency_sentiment_df.drop(columns=["response_count", "data_quality"], errors="ignore"),
            on=["agency_name", "agency_name_display"],
            how="left",
        )
        .merge(
            agency_word_freq_df.drop(columns=["response_count"], errors="ignore"),
            on=["agency_name", "agency_name_display"],
            how="left",
        )
        .merge(
            agency_themes_df.drop(columns=["response_count"], errors="ignore"),
            on=["agency_name", "agency_name_display"],
            how="left",
        )
    )

    # join agency-level competency summary if available
    agency_comp_df = build_agency_competency_summary(evaluations_df)
    if agency_comp_df is not None:
        profiles_df = profiles_df.merge(agency_comp_df, on="agency_name", how="left")
        print(f"  competency summary joined for {agency_comp_df['agency_name'].nunique()} agencies")

    profiles_df = apply_concern_logic(profiles_df)
    profiles_df = apply_misalignment_flag(profiles_df)

    # ----------------------------------------------------------------
    # phase 8: trend analysis and fit scoring
    # ----------------------------------------------------------------
    agency_yearly_trends_df = build_agency_yearly_trends(evaluations_df)
    save_csv(agency_yearly_trends_df, tables_dir, agency_trends_file.name)

    recent_fit_df = build_recent_fit_scores(agency_yearly_trends_df)
    profiles_df = profiles_df.merge(recent_fit_df, on="agency_name", how="left")
    profiles_df = add_fit_trend(profiles_df)

    completeness_df = build_response_completeness(evaluations_df)
    profiles_df = profiles_df.merge(completeness_df, on="agency_name", how="left")

    # ----------------------------------------------------------------
    # phase 9: column ordering and final save
    # ----------------------------------------------------------------
    priority_cols = [
        "agency_name",
        "agency_name_display",
        "response_count",
        "data_quality",
        "composite_fit_score",
        "recent_fit_score",
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
    remaining_cols = [col for col in profiles_df.columns if col not in priority_cols]
    ordered_cols = [col for col in priority_cols if col in profiles_df.columns] + remaining_cols
    profiles_df = profiles_df[ordered_cols].sort_values(
        ["concern_indicator_count", "composite_fit_score", "agency_name_display"],
        ascending=[False, True, True],
    )
    save_csv(profiles_df, profiles_dir, "agency_profiles.csv")

    text_cols_present = ["academic_year", "agency_name", "agency_name_display"] + [
        col for col in text_cols if col in evaluations_df.columns
    ]
    save_csv(evaluations_df[text_cols_present], tables_dir, "evaluations_text.csv")

    # ----------------------------------------------------------------
    # phase 10: summary
    # ----------------------------------------------------------------
    flagged = int((profiles_df["concern_flag"] == "review recommended").sum())
    declining = int((profiles_df["fit_trend"] == "declining").sum())
    misaligned = int((profiles_df.get("misalignment_flag", pd.Series()) == "score-narrative mismatch").sum())
    included = int(profiles_df["most_helpful_top_words"].notna().sum())

    print("step 1: loaded and validated raw data")
    print("step 2: standardized agency names, encoded likert responses, mapped program levels")
    print("step 3: scored sentiment with vader")
    print("step 4: joined program-level competency scores")
    print("step 5: built agency scores, sentiment, word frequency, and theme tables")
    print("step 6: merged into agency profiles, applied concern and misalignment flags")
    print("step 7: built yearly trends, recent fit scores, and trend labels")
    print(f"step 8: {flagged} agencies flagged for review")
    print(f"step 9: {declining} agencies show a declining fit trend")
    print(f"step 10: {misaligned} agencies show a score-narrative misalignment")
    print(
        f"summary: {len(evaluations_df)} evaluations, "
        f"{profiles_df['agency_name'].nunique()} agencies, "
        f"{included} agencies with text-based profile outputs"
    )


if __name__ == "__main__":
    run_pipeline()
    from create_visualizations import generate_all_figures

    generate_all_figures()
