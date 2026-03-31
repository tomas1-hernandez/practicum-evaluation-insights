# practicum evaluation intelligence pipeline
from __future__ import annotations

import os
import re
from collections import Counter

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob

from constants import fit_score_cols, likert_cols, stopwords, text_cols

input_file = "data/raw/feedback_hernandez_20260326_v1.csv"
tables_dir = "outputs/tables"
profiles_dir = "outputs/agency_profiles"
# standardized likert scale
likert_map = {
    "strongly agree": 5,
    "somewhat agree": 4,
    "neither agree nor disagree": 3,
    "somewhat disagree": 2,
    "strongly disagree": 1,
}
# thresholds for agency review flags
min_responses = 3
concern_fit_score = 3.5
concern_sentiment = 0.10
concern_admin_overload = 40.0
concern_supervision_least = 25.0
concern_recommendation = 0.70
concern_threshold = 2

# bigrams that are likely artifacts of the prompt or common phrases
prompt_artifact_bigrams = {
    "least helpful",
    "helpful part",
    "most helpful",
    "helpful educational",
    "helpful least",
    "part least",
    "biggest help",
    "day day",
    "day work",
    "instead observing",
    "one frustration",
    "issues site",
    "needed chances",
    "chances try",
    "try skills",
    "needed client",
    "work enough",
    "enough hands",
    "hands practice",
}

# theme keywords used for simple text tagging
theme_dictionary = {
    "administrative_overload": [
        "administrative",
        "admin",
        "paperwork",
        "shadowing",
        "observing",
        "watched",
        "office tasks",
        "filing",
        "not enough client",
        "too much admin",
        "mostly observing",
        "clerical",
        "data entry",
        "sitting in",
        "just observed",
        "too administrative",
        "too much paperwork",
        "observation only",
        "limited hands",
    ],
    "direct_practice_opportunity": [
        "direct practice",
        "client contact",
        "hands on",
        "hands-on",
        "direct client",
        "caseload",
        "case management",
        "direct service",
        "exposure",
        "real experience",
        "client work",
        "client interaction",
        "practice skills",
        "client sessions",
        "worked with clients",
        "client facing",
        "real cases",
        "real clients",
        "carrying cases",
    ],
    "learning_environment": [
        "learning",
        "learned",
        "educational",
        "confidence",
        "build confidence",
        "assessment",
        "intervention",
        "skills",
        "knowledge",
        "professional development",
        "competency",
        "apply",
        "theories",
        "growth",
        "classroom to practice",
        "put theory",
        "real world",
        "connected theory",
        "skill development",
        "interdisciplinary",
        "multidisciplinary",
        "team meetings",
    ],
    "organizational_structure": [
        "structure",
        "structured",
        "clear expectations",
        "organized",
        "clear direction",
        "unclear",
        "unstructured",
        "disorganized",
        "no structure",
        "lack of structure",
        "well organized",
        "clear goals",
        "expectations",
        "consistent",
        "inconsistent",
        "scheduling",
        "clear boundaries",
        "agency culture",
        "agency policies",
    ],
    "social_justice_alignment": [
        "advocacy",
        "social justice",
        "community",
        "policy",
        "human rights",
        "diversity",
        "equity",
        "empowerment",
        "marginalized",
        "underserved",
        "systemic",
        "macro",
        "social change",
        "oppression",
        "anti-racism",
        "social and economic",
        "policy practice",
        "community organizing",
        "grassroots",
    ],
    "strong_supervision": [
        "supervision",
        "supervisor",
        "supervisory",
        "feedback",
        "strengths-based",
        "mentor",
        "mentoring",
        "guidance",
        "one on one",
        "weekly supervision",
        "timely feedback",
        "regular supervision",
        "supported",
        "felt supported",
        "available",
        "check in",
        "debrief",
        "debriefing",
    ],
}


def clean_text(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.lower().strip()
    value = re.sub(r"[^a-z\s']", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value if len(value) > 2 else None


def normalize_agency_name(value: object) -> str:
    text = str(value).strip().lower().rstrip(".,;")
    return re.sub(r"\s+", " ", text)


def tokenize(value: str | None) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    tokens = re.findall(r"\b[a-z]+\b", value)
    return [token for token in tokens if token not in stopwords and len(token) > 2]


def get_bigrams(tokens: list[str]) -> list[str]:
    bigrams = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
    return [bigram for bigram in bigrams if bigram not in prompt_artifact_bigrams]


def format_counts(counter_obj: Counter, top_n: int) -> str:
    return " | ".join(f"{term}:{count}" for term, count in counter_obj.most_common(top_n))


def save_csv(df: pd.DataFrame, folder: str, filename: str) -> None:
    os.makedirs(folder, exist_ok=True)
    df.to_csv(os.path.join(folder, filename), index=False)


def add_academic_year_start(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["academic_year_start"] = (
        df["academic_year"].astype("string").str.extract(r"(\d{4})").astype(float).astype("Int64")
    )
    return df


def get_agency_display_map(df: pd.DataFrame) -> pd.DataFrame:
    display_df = (
        df.groupby("agency_name")["agency_name_raw"]
        .agg(lambda values: values.astype("string").str.strip().value_counts().idxmax())
        .reset_index()
        .rename(columns={"agency_name_raw": "agency_name_display"})
    )
    return display_df


def build_lda_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for column in ["most_helpful_cleaned", "least_helpful_cleaned"]:
        documents = df[column].dropna().tolist()
        if len(documents) < 10:
            continue

        vectorizer = CountVectorizer(
            max_features=500,
            stop_words="english",
            min_df=5,
            ngram_range=(1, 2),
        )
        matrix = vectorizer.fit_transform(documents)
        lda = LatentDirichletAllocation(
            n_components=6,
            random_state=42,
            learning_method="batch",
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
    rows: list[dict[str, object]] = []

    for agency_name, agency_df in df.groupby("agency_name"):
        if len(agency_df) < min_responses:
            continue

        row = {
            "agency_name": agency_name,
            "response_count": len(agency_df),
            "agency_name_display": agency_df["agency_name_display"].mode().iloc[0],
        }

        for column in ["most_helpful", "least_helpful"]:
            token_counter = Counter()
            bigram_counter = Counter()

            for text in agency_df[f"{column}_cleaned"].dropna():
                tokens = tokenize(text)
                token_counter.update(tokens)
                bigram_counter.update(get_bigrams(tokens))

            row[f"{column}_top_words"] = format_counts(token_counter, 20)
            row[f"{column}_top_bigrams"] = format_counts(bigram_counter, 15)

        rows.append(row)

    return pd.DataFrame(rows)


def add_theme_tags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for theme_name, keywords in theme_dictionary.items():
        df[f"{theme_name}_helpful"] = (
            df["most_helpful_cleaned"].fillna("").apply(lambda text: int(any(keyword in text for keyword in keywords)))
        )
        df[f"{theme_name}_least"] = (
            df["least_helpful_cleaned"].fillna("").apply(lambda text: int(any(keyword in text for keyword in keywords)))
        )
    return df


def build_agency_themes(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for agency_name, agency_df in df.groupby("agency_name"):
        if len(agency_df) < min_responses:
            continue

        row = {
            "agency_name": agency_name,
            "agency_name_display": agency_df["agency_name_display"].mode().iloc[0],
            "response_count": len(agency_df),
        }
        helpful_total = agency_df["most_helpful_cleaned"].notna().sum()
        least_total = agency_df["least_helpful_cleaned"].notna().sum()

        for theme_name in theme_dictionary:
            helpful_matches = agency_df[f"{theme_name}_helpful"].sum()
            least_matches = agency_df[f"{theme_name}_least"].sum()
            row[f"{theme_name}_helpful_pct"] = (
                round(100 * helpful_matches / helpful_total, 1) if helpful_total else None
            )
            row[f"{theme_name}_least_pct"] = round(100 * least_matches / least_total, 1) if least_total else None

        rows.append(row)

    return pd.DataFrame(rows)


def build_grouped_scores(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
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
    polarity_cols = [f"{column}_polarity" for column in text_cols]
    subjectivity_cols = [f"{column}_subjectivity" for column in text_cols]

    grouped = pd.concat(
        [
            df.groupby(group_cols).size().rename("response_count"),
            df.groupby(group_cols)[polarity_cols].mean().round(4),
            df.groupby(group_cols)[subjectivity_cols].mean().round(4),
        ],
        axis=1,
    ).reset_index()

    grouped["overall_sentiment_score"] = grouped[polarity_cols].mean(axis=1).round(4)
    grouped["data_quality"] = grouped["response_count"].apply(
        lambda count: "sufficient" if count >= min_responses else "limited"
    )

    return grouped


def apply_concern_logic(df: pd.DataFrame) -> pd.DataFrame:
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


def build_agency_yearly_trends(evaluations_df: pd.DataFrame) -> pd.DataFrame:
    trend_scores = build_grouped_scores(
        evaluations_df,
        ["academic_year", "agency_name", "agency_name_display"],
    )

    trend_sentiment = build_grouped_sentiment(
        evaluations_df,
        ["academic_year", "agency_name", "agency_name_display"],
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

    trends_df = trends_df[priority_cols + remaining_cols].sort_values(["academic_year_start", "agency_name_display"])

    return trends_df


def run_pipeline() -> None:
    print("running practicum evaluation intelligence pipeline")

    evaluations_df = pd.read_csv(input_file)
    evaluations_df = add_academic_year_start(evaluations_df)

    evaluations_df["agency_name_raw"] = evaluations_df["agency_name"].astype("string").str.strip()
    evaluations_df["agency_name"] = evaluations_df["agency_name_raw"].apply(normalize_agency_name)
    display_map_df = get_agency_display_map(evaluations_df)
    evaluations_df = evaluations_df.merge(display_map_df, on="agency_name", how="left")

    for column in likert_cols:
        evaluations_df[column] = evaluations_df[column].astype("string").str.strip().str.lower().map(likert_map)

    evaluations_df["recommend_num"] = (
        evaluations_df["recommend"].astype("string").str.strip().str.lower().map({"yes": 1, "no": 0})
    )

    for column in text_cols:
        evaluations_df[f"{column}_cleaned"] = evaluations_df[column].apply(clean_text)
        sentiment = evaluations_df[column].apply(
            lambda value: TextBlob(value).sentiment if isinstance(value, str) and value.strip() else None
        )
        evaluations_df[f"{column}_polarity"] = sentiment.apply(
            lambda value: round(value.polarity, 4) if value is not None else None
        )
        evaluations_df[f"{column}_subjectivity"] = sentiment.apply(
            lambda value: round(value.subjectivity, 4) if value is not None else None
        )

    agency_scores_df = build_grouped_scores(evaluations_df, ["agency_name", "agency_name_display"])
    save_csv(agency_scores_df.sort_values("agency_name_display"), tables_dir, "agency_scores.csv")

    agency_sentiment_df = build_grouped_sentiment(evaluations_df, ["agency_name", "agency_name_display"])
    save_csv(agency_sentiment_df.sort_values("agency_name_display"), tables_dir, "agency_sentiment.csv")

    lda_topics_df = build_lda_summary(evaluations_df)
    save_csv(lda_topics_df, tables_dir, "lda_topics.csv")

    agency_word_freq_df = build_word_freq_summary(evaluations_df)
    save_csv(agency_word_freq_df.sort_values("agency_name_display"), tables_dir, "agency_word_freq.csv")

    evaluations_df = add_theme_tags(evaluations_df)
    agency_themes_df = build_agency_themes(evaluations_df)
    save_csv(agency_themes_df.sort_values("agency_name_display"), tables_dir, "agency_themes.csv")

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

    profiles_df = apply_concern_logic(profiles_df)

    priority_cols = [
        "agency_name",
        "agency_name_display",
        "response_count",
        "data_quality",
        "composite_fit_score",
        "recommendation_rate",
        "overall_sentiment_score",
        "concern_indicator_count",
        "concern_flag",
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
    profiles_df = profiles_df[[col for col in priority_cols if col in profiles_df.columns] + remaining_cols]
    profiles_df = profiles_df.sort_values(
        ["concern_indicator_count", "composite_fit_score", "agency_name_display"], ascending=[False, True, True]
    )
    save_csv(profiles_df, profiles_dir, "agency_profiles.csv")

    agency_yearly_trends_df = build_agency_yearly_trends(evaluations_df)
    save_csv(agency_yearly_trends_df, tables_dir, "agency_yearly_trends.csv")

    text_cols_present = ["academic_year", "agency_name", "agency_name_display"] + [
        c for c in text_cols if c in evaluations_df.columns
    ]
    save_csv(evaluations_df[text_cols_present], tables_dir, "evaluations_text.csv")

    flagged = int((profiles_df["concern_flag"] == "review recommended").sum())
    included = int(profiles_df["most_helpful_top_words"].notna().sum())

    print("step 1: processed and profiled agencies")
    print("step 2: saved agency tables and profiles")
    print(f"step 3: {flagged} agencies met the current review threshold")
    print(
        f"summary: {len(evaluations_df)} evaluations, "
        f"{profiles_df['agency_name'].nunique()} agencies, "
        f"{included} agencies with text-based profile outputs"
    )


if __name__ == "__main__":
    run_pipeline()
    from create_visualizations import generate_all_figures
    generate_all_figures()
