# Practicum Evaluation Intelligence

Practicum Evaluation Intelligence is a Python workflow that turns end-of-practicum student evaluations into agency profiles, concern flags, trend views, report figures, descriptive tables, and a leadership-facing dashboard.

Social work programs must evaluate the effectiveness of their field settings under CSWE's 2022 accreditation standards, including collecting end-of-practicum evaluation data each year. This project turns those evaluations into a clearer review system that helps field education teams identify patterns, support earlier follow-up, and monitor agency quality over time.

## Project links

- [Streamlit dashboard](https://umssw-placement-quality-dashboard.streamlit.app)
- [Google Colab Walkthrough](https://colab.research.google.com/drive/1llw5HmqDmM9jRuJpVPn5R9kdvfHo3xU-?usp=sharing)
- [Digital Portfolio](https://ssw-peit.netlify.app/)
- [Written report](https://umconnectumt-my.sharepoint.com/:w:/g/personal/tomas_hernandez_umt_edu/IQDI2uH3Lo99SLwfoRKfm5CUAUJ1QJJqtv6eDaGAewhi9Yw?e=4Gi9SH)

## What the data shows

Using 2,258 evaluations from 222 agencies across 12 academic years, the workflow found:

- 64 agencies met the threshold for leadership review by showing two or more overlapping concern signals
- 68 agencies showed a declining Placement Quality Score before reaching the full flag threshold
- the relationship between CSWE competency benchmark scores and Placement Quality Scores was essentially zero across 217 agencies, r = -0.009, p = 0.89
- 2 agencies showed strong benchmark scores while students described some of the weakest placement experiences in the dataset
- 2025 to 2026 was the first year the program-wide Placement Quality Score fell below the 3.5 concern threshold
- MSW students rated placements higher than BSW students on all six measures analyzed, with three differences reaching statistical significance

## What this project does

This workflow:

- reads and cleans the evaluation data
- scores key placement items into a Placement Quality Score
- calculates recommendation rates
- scores open-ended comments with VADER sentiment
- tags recurring themes in student comments
- builds one-row agency profiles
- builds agency-by-year trend tables
- applies multi-signal review logic
- generates report figures and summary tables
- feeds a Streamlit dashboard for interactive review

## Quick start

Clone the repo and run the full workflow from the project root.

```bash
git clone https://github.com/tomas1-hernandez/practicum-evaluation-intelligence.git
cd practicum-evaluation-intelligence

python -m venv .venv
```
Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Install the requirements and run the workflow:

```bash
pip install -r requirements.txt
python run_all.py
```

To open the dashboard:

```bash
streamlit run app.py
```

## Workflow order

The full workflow runs these scripts in order:

1. `pipeline.py`
2. `create_visualizations.py`
3. `analyze_program_alignment.py`
4. `build_descriptive_tables.py`

`run_all.py` runs them in that order for you.

## Main files

* `run_all.py` - runs the full workflow
* `pipeline.py` - core processing and output generation
* `create_visualizations.py` - static report figures
* `analyze_program_alignment.py` - BSW/MSW comparison and competency alignment analysis
* `build_descriptive_tables.py` - summary tables for EDA and reporting
* `app.py` - Streamlit dashboard
* `config.py` - file paths, thresholds, and shared settings
* `constants.py` - shared labels, mappings, and reference values
* `theme_lexicon.py` - keyword dictionary for theme tagging

## Main outputs

### `outputs/agency_profiles/agency_profiles.csv`

Below is the main agency-level output. It combines:

* Placement Quality Score
* recommendation rate
* sentiment score
* trend label
* concern indicator count
* concern flag
* misalignment flag
* theme percentages
* top phrases
* response volume

### `outputs/tables/agency_yearly_trends.csv`

One row per agency per academic year. Used for trend monitoring and dashboard views.

### `outputs/figures/`

Static PNG figures used in the report, portfolio, and presentation materials.

### `outputs/tables/`

Saved summary tables, including descriptive tables, yearly trends, program split outputs, and competency alignment outputs.

## Review logic

An agency is flagged for review only when two or more concern signals occur at the same time.

The current concern signals are:

* Placement Quality Score below 3.5
* sentiment score below 0.05
* recommendation rate below 70%
* administrative overload above 40% of least-helpful responses
* supervision concern above 25% of least-helpful responses

This keeps one weak metric from acting like a final verdict and makes the flag a structured prompt for follow-up.

## Dashboard

The Streamlit dashboard lets users:

* filter agencies by concern status, trend, score range, and response count
* review flagged agencies more quickly
* compare scores, themes, phrases, and yearly trends
* inspect agency-level outputs without rerunning the code

The dashboard reads the saved CSV outputs. Rerun `python run_all.py` any time the data changes.

## Using your own data

If your program collects end-of-practicum evaluations with agency names, Likert items, open-ended comments, and a recommendation item, this workflow can be adapted to work with your data.

Basic steps:

1. replace the files in `data/`
2. update file paths in `config.py` if needed
3. update column mappings in `constants.py` if needed
4. review thresholds in `config.py`
5. run `python run_all.py`

## Requirements

Main packages used:

* pandas
* numpy
* matplotlib
* seaborn
* plotly
* streamlit
* vaderSentiment
* wordcloud
* scikit-learn
* scipy

Install everything with:

```bash
pip install -r requirements.txt
```

Python 3.10 or higher is recommended.

## Reproducibility

Running the workflow on the same input files produces the same outputs.

The Colab notebook mirrors the public repo workflow, and the dashboard reads the saved outputs generated by the pipeline.

## Author

**Tomas Hernandez**  
MSBA Capstone, BMKT 699  
University of Montana  
Spring 2026  
