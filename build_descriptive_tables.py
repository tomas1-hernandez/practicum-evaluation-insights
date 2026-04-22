"""Build report-ready descriptive statistics tables."""

import pandas as pd

from config import agency_profiles_file, input_file, tables_dir
from constants import likert_cols, program_level_map, text_cols


def describe_series(series: pd.Series) -> dict[str, float | int]:
    """Return compact descriptive stats for one numeric series."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "n": int(s.count()),
        "mean": round(s.mean(), 3),
        "sd": round(s.std(), 3),
        "min": round(s.min(), 3),
        "median": round(s.median(), 3),
        "max": round(s.max(), 3),
    }


def build_dataset_overview(
    raw_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the dataset overview table used in the report."""
    program_series = (
        raw_df["program_year"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map(program_level_map)
    )

    bsw_count = int((program_series == "bsw").sum())
    msw_count = int((program_series == "msw").sum())
    total = len(raw_df)
    sufficient_count = int((profiles_df["data_quality"] == "sufficient").sum())

    return pd.DataFrame(
        {
            "Measure": [
                "Student evaluations",
                "Practicum agencies",
                "Academic years",
                "Agencies with sufficient data for profiling",
                "BSW responses",
                "BSW share",
                "MSW responses",
                "MSW share",
                "Structured Likert items",
                "Open-ended response fields",
            ],
            "Value": [
                total,
                raw_df["agency_name"].nunique(),
                raw_df["academic_year"].nunique(),
                sufficient_count,
                bsw_count,
                f"{round((bsw_count / total) * 100, 1)}%",
                msw_count,
                f"{round((msw_count / total) * 100, 1)}%",
                len(likert_cols),
                len(text_cols),
            ],
        }
    )


def build_numeric_descriptives(profiles_df: pd.DataFrame) -> pd.DataFrame:
    """Build descriptive stats for the main agency-level variables."""
    profiles_df = profiles_df.copy()
    profiles_df["recommendation_rate_pct"] = (
        pd.to_numeric(profiles_df["recommendation_rate"], errors="coerce")
        * 100
    )

    variables = {
        "Placement Quality Score": "placement_quality_score",
        "Recommendation rate (%)": "recommendation_rate_pct",
        "Sentiment score": "overall_sentiment_score",
        "Response count": "response_count",
        "Concern indicator count": "concern_indicator_count",
        "Mean competency score": "mean_competency_score",
    }

    rows = []
    for label, col in variables.items():
        stats = describe_series(profiles_df[col])
        stats["Variable"] = label
        rows.append(stats)

    return pd.DataFrame(rows)[
        ["Variable", "n", "mean", "sd", "min", "median", "max"]
    ]


def build_program_split_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build a small BSW/MSW count table."""
    program_series = (
        raw_df["program_year"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map(program_level_map)
    )

    split_df = pd.DataFrame(
        {
            "Program level": ["BSW", "MSW"],
            "Count": [
                int((program_series == "bsw").sum()),
                int((program_series == "msw").sum()),
            ],
        }
    )

    total = split_df["Count"].sum()
    split_df["Share"] = (split_df["Count"] / total * 100).round(1).astype(
        str
    ) + "%"

    return split_df


def main() -> None:
    """Create descriptive tables and save them to outputs/tables."""
    tables_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(input_file)
    profiles_df = pd.read_csv(agency_profiles_file)

    dataset_overview_df = build_dataset_overview(raw_df, profiles_df)
    descriptive_stats_df = build_numeric_descriptives(profiles_df)
    program_split_df = build_program_split_summary(raw_df)

    dataset_overview_df.to_csv(
        tables_dir / "dataset_overview.csv", index=False
    )
    descriptive_stats_df.to_csv(
        tables_dir / "descriptive_stats_main.csv",
        index=False,
    )
    program_split_df.to_csv(
        tables_dir / "program_split_summary.csv",
        index=False,
    )

    print("saved dataset_overview.csv")
    print("saved descriptive_stats_main.csv")
    print("saved program_split_summary.csv")


if __name__ == "__main__":
    main()
