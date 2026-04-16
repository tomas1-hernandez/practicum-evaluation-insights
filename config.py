"""Project settings for practicum evaluation intelligence.

This file keeps the main paths and decision rules in one place so the rest of
the project does not have hard-coded values scattered everywhere. If the data
file, output folders, or review thresholds ever need to change, this is the
first place to look.

The values here are written for the current umssw capstone project. They can be
updated later if another social work program wants to adapt the pipeline to
their own evaluation data.
"""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parent

data_dir = project_root / "data"

# primary evaluation export - one row per student response
input_file = data_dir / "feedback_hernandez_20260326_v1.csv"

# program-level competency scores by academic year
# rows: one per program_level + academic_year combination
# columns: program_level, academic_year, competency_1 through competency_9
competency_file = data_dir / "competency_scores_by_year.csv"

outputs_dir = project_root / "outputs"
figures_dir = outputs_dir / "figures"
profiles_dir = outputs_dir / "agency_profiles"
tables_dir = outputs_dir / "tables"

agency_profiles_file = profiles_dir / "agency_profiles.csv"
agency_trends_file = tables_dir / "agency_yearly_trends.csv"
evaluations_text_file = tables_dir / "evaluations_text.csv"

# minimum student responses for an agency to be included in profile outputs
min_responses = 3

# concern flag thresholds - an agency is flagged when concern_threshold or more
# of the indicators below fire at the same time
concern_fit_score = 3.5        # placement quality score below this
concern_sentiment = 0.05       # overall vader sentiment below this
concern_admin_overload = 40.0  # pct of least_helpful responses tagged admin overload
concern_supervision_least = 25.0  # pct of least_helpful responses tagged supervision
concern_recommendation = 0.70  # recommendation rate below this
concern_threshold = 2          # minimum simultaneous indicators to trigger a flag

# misalignment flag - fires when program competency scores are strong but
# the agency's own student narrative signals tell a different story
# a high program score paired with low agency sentiment or high admin overload
# suggests students are passing benchmarks but not experiencing what they describe
misalignment_comp_floor = 85.0    # program competency average above this is "strong"
misalignment_sentiment_ceiling = 0.10  # agency sentiment below this despite strong comps
misalignment_admin_ceiling = 35.0  # admin overload pct above this despite strong comps

# trend flag - how far recent fit score must drop below all-years average
declining_threshold = 0.30

figure_dpi = 150

app_title = "practicum evaluation intelligence"
