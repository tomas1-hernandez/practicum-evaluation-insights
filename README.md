# Practicum Evaluation Insights

**A review workflow that helps field education teams monitor practicum agency quality over time.**

Field education is where social work begins. This keeps it strong.

Practicum Evaluation Insights is a Python workflow that turns end-of-practicum student evaluations into agency profiles, review flags, trend views, report figures, and an interactive Streamlit dashboard. This system was created for social work education programs that gather student feedback each year. It helps clarify trends in agencies over time.



## Overview

This project grew out of field education work and the yearly review of end-of-practicum evaluations.

This project emerged from field education work and the annual review of practicum evaluations. While the yearly process was useful, it was challenging to clearly identify the agency's history over time. Although feedback could be summarized and stored, comparing patterns across agencies was difficult, especially since students often provide their best feedback after their practicum ends. 

Practicum Evaluation Insights was developed to address this issue. It does not judge or label agencies but facilitates a clearer understanding of placement history for the Field Education Team, helping to detect repeated issues earlier and ensuring consistent follow-up.



## Why This Repository Exists

Field education is the signature pedagogy of social work. CSWE’s 2022 Educational Policy and Accreditation Standards require accredited social work programs to evaluate the effectiveness of field-based settings, monitor student learning, and use field education data for continuous improvement.

This repository outlines a review process that can be:  

- Rerun annually
- Adapted for other program evaluations
- Handed off to future staff or students
- Maintained without ongoing support from the original author



## What Gets Missed Without a Review Process

Students often hold back concerns during practicum to maintain relationships or avoid conflict. By the time the end-of-practicum evaluation provides explicit feedback, the placement has already ended.

Without a review process:

- A low recommendation rate may seem like an isolated issue, masking a larger pattern.
- Similar supervision concerns may recur across years.
- An agency might seem acceptable on paper, while student feedback reveals flaws.
- Multiple students may experience the same weak placement before patterns are recognized.

This workflow addresses the problem.



## Project Links

| Resource | Link |
|---|---|
| Streamlit dashboard | https://umssw-placement-quality-dashboard.streamlit.app |
| Google Colab walkthrough | https://colab.research.google.com/drive/1llw5HmqDmM9jRuJpVPn5R9kdvfHo3xU-?usp=sharing |
| Digital portfolio | https://ssw-peit.netlify.app |




## What the Workflow Does

The workflow consists of five stages:

1. **Import evaluation data**
   - Loads primary evaluation exports and competency benchmarks.

2. **Clean and standardize records**
   - Standardizes agency names, aligns program values, validates fields, and converts Likert responses to numeric values.

3. **Score structured and narrative signals**
   - Builds Placement Quality Score, recommendation rate, sentiment score, and theme-based reviews.

4. **Build agency outputs**
   - Creates an agency profile per site and trends file for changes over time.

5. **Save review outputs**
   - Produces CSV files, static figures, and dashboard-ready files for review.

The workflow does not make final decisions for people. It produces structured review signals that the Field Education Team can interpret alongside agency history, site context, and program knowledge.



## What the Data Shows

Analysis of 2,258 evaluations from 222 practicum agencies over 12 years revealed:

- **64 agencies** showed two or more overlapping concern signals, qualifying for leadership review.
- **68 agencies** exhibited a declining Placement Quality Score before hitting the review flag threshold.
- **2 agencies** demonstrated strong program-level competencies, yet students reported poor placement experiences.
- **r = -0.009, p = 0.89** indicates no significant relationship between competency benchmarks and Placement Quality Scores across 217 agencies.
- **2025-2026** marked the first year the program-wide Placement Quality Score fell below the 3.5 concern threshold.
- **MSW students** rated placements higher than **BSW students** on all measures, with three differences statistically significant.

These results highlight that competency success and placement quality assess different aspects and are not interchangeable.



## For Non-Technical Users

The Google Colab notebook is the easiest entry point for users who want to run the workflow without installing Python locally. It runs entirely in a browser.

1. Open the [Colab notebook](https://colab.research.google.com/drive/1llw5HmqDmM9jRuJpVPn5R9kdvfHo3xU-?usp=sharing)
2. Click **Runtime > Run all**
3. The notebook clones the repository, installs requirements, runs the full pipeline, and displays the outputs
4. Download the outputs as a zip file from the final cell

No Python installation is required. No command line is required. Any staff member with a Google account can run it and get the same results.



## Quick Start for Local Installation

Clone the repository and set up a virtual environment.

```bash
git clone https://github.com/tomas1-hernandez/practicum-evaluation-insights.git
cd practicum-evaluation-insights
python -m venv .venv
```

Activate the environment.

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Install requirements and run the full workflow.

```bash
pip install -r requirements.txt
python run_all.py
```

Open the dashboard.

```bash
streamlit run app.py
```

Python 3.10 or higher is recommended.



## Recommended Use Order

For a new user or a future handoff, the best order is:

1. Read this README
2. Review `config.py`
3. Review `constants.py`
4. Add the required input files to `data/`
5. Run `python run_all.py`
6. Open the dashboard with `streamlit run app.py`
7. Review the saved outputs in `outputs/`



## Workflow Order

`run_all.py` runs these scripts in order:

| Step | Script | What it does |
|---|---|---|
| 1 | `pipeline.py` | Cleans, scores, tags, flags, and aggregates all evaluations into agency profiles and yearly trends |
| 2 | `create_visualizations.py` | Generates all static report figures as PNG files |
| 3 | `analyze_program_alignment.py` | Runs BSW vs MSW comparison and competency alignment analysis |
| 4 | `build_descriptive_tables.py` | Builds EDA summary tables for the report |

Run components individually as needed or execute the full workflow with `python run_all.py`.



## Repository Structure

```text
practicum-evaluation-intelligence/
|
|-- run_all.py
|-- pipeline.py
|-- create_visualizations.py
|-- analyze_program_alignment.py
|-- build_descriptive_tables.py
|-- app.py
|-- config.py
|-- constants.py
|-- theme_lexicon.py
|-- requirements.txt
|-- README.md
|
|-- data/
|   |-- feedback_hernandez_20260326_v1.csv
|   |-- competency_scores_by_year.csv
|
|-- outputs/
    |-- agency_profiles/
    |   |-- agency_profiles.csv
    |-- tables/
    |   |-- agency_yearly_trends.csv
    |   |-- evaluations_text.csv
    |   |-- program_split_summary.csv
    |   |-- competency_alignment_summary.csv
    |-- figures/
        |-- [static PNG figures]
```



## Input Data Requirements

### Primary evaluation file

The primary evaluation file should contain one row per student response.

Required columns at minimum:

| Column | Type | Description |
|---|---|---|
| `agency_name` | string | Practicum agency name |
| `academic_year` | string | Academic year label, for example `2024-2025` |
| `program_year` | string | Student level, such as `bsw`, `generalist`, or `specialization` |
| `recommend` | string | `yes` or `no` |
| `most_helpful` | string | Open-ended response |
| `least_helpful` | string | Open-ended response |
| `prepared_for_practice` | string | Likert item |
| `learning_goals_met` | string | Likert item |
| `supervision_frequency` | string | Likert item |
| `supervision_quality` | string | Likert item |
| `felt_prepared` | string | Likert item |
| `comp_1` ... `comp_9` | string | Nine competency items |

Additional open-ended fields, when available, are also used:

- `recommend_reason`
- `comments_supervisor`
- `other_comments`

Valid Likert responses:

- `strongly disagree`
- `somewhat disagree`
- `neither agree nor disagree`
- `somewhat agree`
- `strongly agree`

If your column names differ, update the mappings in `constants.py` before running the workflow.

### Competency scores file

The competency scores file should contain one row per academic year and program-level combination.

| Column | Type | Description |
|---|---|---|
| `program_level` | string | `BSW` or `MSW` |
| `academic_year` | string | Academic year label matching the evaluation file |
| `competency_1` ... `competency_9` | float | Program-level benchmark scores from 0 to 100 |



## Outputs

All outputs are written to the `outputs/` folder.

### `outputs/agency_profiles/agency_profiles.csv`

One row per agency. This is the main output used by the dashboard and for field education leadership review.

| Column | Description |
|---|---|
| `agency_name_display` | Standardized agency name |
| `placement_quality_score` | Mean of six Likert items, scored 1 to 5 |
| `recommendation_rate` | Share of students who would recommend the agency |
| `overall_sentiment_score` | VADER compound score on open-ended responses |
| `concern_indicator_count` | Number of active concern signals |
| `concern_flag` | `Review recommended` or `No flag` |
| `fit_trend` | `Improving`, `Stable`, or `Declining` |
| `misalignment_flag` | `Score-narrative mismatch` or `No mismatch` |
| `response_count` | Number of evaluations included |
| `data_quality` | `Sufficient` or `Insufficient` |

### `outputs/tables/agency_yearly_trends.csv`

One row per agency per academic year. Used for trend monitoring and dashboard time-series views.

Key fields include:

- Yearly Placement Quality Score
- Yearly recommendation rate
- Yearly sentiment score
- Yearly review flag
- Yearly response count

### `outputs/figures/`

Static PNG figures for the written report, portfolio, and presentation. Generated by `create_visualizations.py`.

### `outputs/tables/`

Descriptive tables, program-level split outputs, competency alignment outputs, and supporting summaries.



## How the Flag Logic Works

An agency is flagged for leadership review only when **two or more concern signals are active at the same time**. Requiring multiple signals keeps a single weak score from acting as a verdict and keeps the flag focused on agencies where multiple aspects of the student experience are weaker at the same time.

### The five concern signals

| Signal | Threshold | Rationale |
|---|---|---|
| Placement quality score | Below 3.5 | Marks the lower portion of the distribution where scores begin to meaningfully separate from the bulk of agencies |
| Recommendation rate | Below 70% | Indicates a meaningful share of students would not recommend the site to a future student |
| Sentiment score | Below 0.05 | Captures language trending toward neutral or negative in professional evaluation writing |
| Administrative overload | Above 40% of least-helpful responses | Students describe the placement as paperwork-heavy or disconnected from meaningful practice |
| Supervision concerns | Above 25% of least-helpful responses | Students describe recurring concerns with supervision quality or support |

All thresholds are stored in `config.py` and can be adjusted without modifying the pipeline itself. They are starting points for review, not fixed standards.

### The trend layer

The pipeline also identifies agencies whose recent Placement Quality Score has dropped more than 0.30 points below their own historical average. This provides an early-warning layer for agencies that have not yet reached the full review flag threshold but may be heading in the wrong direction.

Trend data should be interpreted alongside institutional context. Agencies that have undergone mergers, staffing transitions, or other organizational changes may show apparent declines that reflect disruption rather than a lasting shift in placement quality.



## Dashboard

The Streamlit dashboard reads the saved CSV outputs and provides six interactive views:

- **Overview**: program-wide KPIs, concern scatter, and multi-year trend
- **Flagged Agencies**: ranked list with scores and concern signals
- **Trend Monitoring**: agency-by-agency and program-wide trends
- **Language Patterns**: theme frequency views and phrase clouds
- **Agency Profile**: drill-down view with scores, themes, phrases, and trend
- **Benchmark Alignment**: competency scores compared with Placement Quality Scores

To refresh the dashboard after a new evaluation cycle, run `python run_all.py` and restart the dashboard. The dashboard itself requires no code changes between cycles.



## Responsible Interpretation

This workflow is a review tool, not a decision engine.

A review flag means something is worth a closer look. It does not explain why a pattern exists, and it should not trigger an automatic consequence on its own.

Field context still matters. Agency changes, student needs, supervision transitions, employment-based placements, and local history all shape how a pattern should be understood.

The workflow helps the Field Education Team:

1. Review the agency profile
2. Check trend and review signals
3. Bring in local context
4. Decide whether follow-up is needed

That sequence is intentional. The workflow supports judgment. It does not replace it.



## Adapting This Workflow for Another Program

This workflow can be adapted if your program collects end-of-practicum evaluations with the following components:

1. **Agency identifier**: a consistent name or code for each practicum site
2. **Likert-rated items**: structured ratings on supervision, learning goals, and readiness
3. **Open-ended responses**: student comments in text fields
4. **Recommendation item**: a yes/no or scaled item asking whether students would recommend the site

The logic does not depend on UMSSW-specific fields. Programs using Qualtrics, Sonia, Experiential Learning Cloud, or any consistent spreadsheet format can adapt the pipeline.

### Steps to adapt

1. Place your evaluation export in `data/`
2. Open `config.py` and update `DATA_FILE` to point to your file
3. Open `constants.py` and update the column name mappings to match your instrument
4. Review the concern thresholds in `config.py` against your own data distributions
5. Adjust thresholds if needed
6. Run `python run_all.py`

For many programs, the main setup work is front-loaded. Once the first mapping is complete, yearly updates are a matter of adding the next file and rerunning the workflow.



## Annual Update Process

At the end of each evaluation cycle:

1. Export the new evaluation responses from your survey platform
2. Add the new file to `data/` or append it to the existing data file
3. Run `python run_all.py` from the project root
4. Restart the dashboard with `streamlit run app.py`

The pipeline is reproducible. Running it on the same input files produces the same outputs.



## Limitations

- Agencies with fewer than three evaluations in a given year are excluded from that year's analysis. Small cohorts can produce unstable scores that do not reflect a durable pattern.
- The pipeline identifies patterns, not causes. A concern flag means multiple signals are active at the same time. It does not explain why, and it is not a verdict.
- Sentiment analysis uses VADER, a general-purpose tool. It handles short evaluative comments well but may misread field-specific language in some cases.
- The theme tagging lexicon was developed for social work evaluation language. Programs with meaningfully different comment styles may need to revise or expand `theme_lexicon.py`.
- Competency scores used in the alignment analysis are program-level aggregates joined to agency evaluations. Individual-level competency data was not available.



## Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

Main packages include:

| Package | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, and aggregation |
| `numpy` | Numerical operations |
| `matplotlib` | Static report figures |
| `seaborn` | Statistical visualization |
| `plotly` | Interactive chart components |
| `streamlit` | Dashboard interface |
| `vaderSentiment` | Sentiment scoring on open-ended text |
| `wordcloud` | Word cloud figures |
| `scikit-learn` | Supporting analysis utilities |
| `scipy` | Statistical tests |

Python 3.10 or higher is recommended.



## Reproducibility

To reproduce this project from scratch:

1. Clone the repository
2. Install requirements
3. Place the evaluation data file in `data/`
4. Run `python run_all.py`
5. Run `streamlit run app.py`

All outputs are generated from the workflow and saved to the `outputs/` folder. The dashboard reads saved CSV outputs. If the data changes, rerun the workflow before opening the dashboard again.



## Handoff and Maintenance Notes

This repository is meant to be maintainable by a technically capable staff member, analyst, or future graduate student.

To support handoff:

- The pipeline is modular
- Thresholds are centralized in `config.py`
- Column mappings are centralized in `constants.py`
- The dashboard reads saved outputs rather than rerunning the pipeline
- The Google Colab notebook mirrors the public repository flow
- The README explains both project logic and technical setup

If you are inheriting this repository, start by reading:

1. `README.md`
2. `config.py`
3. `run_all.py`
4. `pipeline.py`



## Author

**Tomas Hernandez**  
MSBA Capstone, BMKT 699  
University of Montana  
Spring 2026

