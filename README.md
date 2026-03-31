# Practicum Evaluation Intelligence

This project is a leadership-facing analytics pipeline for social work field education. It uses end-of-practicum student evaluations to build agency-level profiles, flag agencies that may need a closer look, summarize common strengths and frustrations in student feedback, and track trends over time.



## Table of Contents

1. [Project Overview](#project-overview)
2. [The Problem](#the-problem)
3. [Methods](#methods)
4. [Concern Flag Logic](#concern-flag-logic)
5. [Data and Privacy](#data-and-privacy)
6. [Source Files and Logic](#source-files-and-logic)
7. [Key Outputs](#key-outputs)
8. [Setup and Installation](#setup-and-installation)
9. [Running the Pipeline](#running-the-pipeline)
10. [Streamlit Dashboard](#streamlit-dashboard)
11. [Ethical Considerations](#ethical-considerations)
12. [Limitations](#limitations)
13. [Who This Is For](#who-this-is-for)



## Project Overview

Social work practicum programs collect detailed end-of-placement evaluations from students every year. These evaluations include structured Likert-scale ratings, yes/no recommendation questions, and multiple open-ended text responses about supervision quality, learning opportunities, and overall experience.

In many programs, this data is reviewed one placement at a time and then stored away. Patterns across agencies and years are rarely surfaced in a consistent way. This project builds a pipeline that processes 12 years of evaluation data, 2,258 responses across 222 practicum agencies, and turns it into agency-level profiles that can better support placement review, agency partnerships, and program improvement.

This system is meant to support leadership review, not replace professional judgment. The results are presented as overall trends to identify areas that may need further attention.



## The Problem

Practicum evaluations often contain useful and honest feedback about supervision quality, direct practice opportunities, organizational structure, and student readiness. The challenge is that this feedback usually arrives at the end of a placement, after the experience is already over.

By the time patterns become obvious through manual review, it is too late to help the student who has already completed the placement. This project asks whether evaluation data can be processed more systematically so leadership can spot recurring strengths and concern patterns earlier and use that information to support better future placements.



## Methods

*   **Likert encoding:** Text responses from strongly agree to strongly disagree are converted to a numeric scale from 5 to 1 for quantitative summaries.
*   **Sentiment analysis:** TextBlob polarity scoring is applied to five open-ended text columns. Scores range from -1.0 for more negative language to +1.0 for more positive language.
*   **Word and bigram frequency:** Custom tokenization is used with a social-work-aware stopword list. Single words and consecutive two-word phrases are extracted at both the agency and overall level.
*   **LDA topic modeling:** Latent Dirichlet Allocation with six topics is run separately on the `most_helpful` and `least_helpful` columns using scikit-learn.
*   **Dictionary-based theme tagging:** A custom theme dictionary is used to tag common practicum themes in the text. The six main themes are Strong Supervision, Direct Practice Opportunity, Administrative Overload, Organizational Structure, Learning Environment, and Social Justice Alignment.
*   **Yearly trend analysis:** Agency-level results are also summarized by academic year so leadership can review changes in fit score, recommendation rate, sentiment, and concern flags over time.



## Concern Flag Logic

An agency is marked as **Review Recommended** when it triggers 2 or more of the following indicators simultaneously. This approach helps leadership focus on high-risk placements that need intervention.

| Indicator                   | Threshold |
| :-- | :-- |
| Composite Fit Score         | < 3.5     |
| Overall Sentiment Score     | < 0.10    |
| Administrative Overload     | >= 40%    |
| Supervision Concern         | >= 25%    |
| Recommendation Rate         | < 70%     |

In the current dataset, 66 of 217 agencies included in the text-based profile outputs met this threshold.



## Data & Privacy

* **Evaluations:** 2,258
* **Agencies:** 222 in the raw data, 217 included in the text-based profile outputs
* **Academic Years:** 2014-2015 through 2025-2026
* **Privacy and FERPA:** To protect student identity and support FERPA compliance, personally identifiable information is redacted. The raw data file is not included in this repository.



## Source Files and Logic

This project is organized into separate Python scripts to keep the workflow easier to follow.

* **`src/pipeline.py`** - runs the full pipeline from raw data to saved outputs
* **`src/create_visualizations.py`** - builds the 10 PNG figures from the saved pipeline outputs
* **`src/app.py`** - runs the Streamlit dashboard for filtering and reviewing agency profiles, flags, themes, and yearly trends
* **`src/constants.py`** - stores shared constants used across the project, including thresholds, mappings, and repeated settings

## Key Outputs

A successful run creates the main output files in the `outputs/` folder.

### Agency Reports

* **`outputs/agency_profiles/agency_profiles.csv`** - the main output table with one row per agency, including scores, sentiment, themes, text summaries, and concern flag
* **`outputs/tables/agency_yearly_trends.csv`** - yearly trend output by agency and academic year

### Qualitative Themes

* **`outputs/tables/agency_themes.csv`** - summarizes the percentage of responses tied to themes such as administrative overload, direct practice opportunity, learning environment, organizational structure, social justice alignment, and strong supervision
* **`outputs/tables/lda_topics.csv`** - topic modeling output based on common keywords in the text data

### Visualizations

* **`outputs/figures/`** - includes the 10 PNG figures used in reporting and presentations, including score summaries, concern flag views, word clouds, and yearly trend charts



## Setup and Installation

**Prerequisites**

* Python 3.9+
* Core Dependencies:

  * `pandas`
  * `scikit-learn`
  * `textblob`
  * `matplotlib`
  * `plotly`
  * `wordcloud`
  * `streamlit`

**Installation**

1. Clone the repository

```bash
git clone https://github.com
cd practicum-evaluation-intelligence
```

2. Install dependencies

```bash
pip install -r requirements.txt
```



## Running the Pipeline

The pipeline handles data cleaning, scoring, sentiment analysis, theme summaries, yearly trends, and the generation of 10 reporting figures.

```bash
python src/pipeline.py
```

**Pipeline steps** 

1. Read the raw evaluation CSV
2. Normalize agency names, score Likert responses, and run sentiment analysis on text fields
3. Build agency-level scores, sentiment summaries, theme tags, word frequency tables, and yearly trend data
4. Apply concern flag logic and build the master agency profiles table
5. Save output tables to `outputs/tables/` and agency profiles to `outputs/agency_profiles/`
6. Generate figures 1 through 10 and save them to `outputs/figures/`

**Expected output on a successful run**

```text
running practicum evaluation intelligence pipeline
step 1: processed and profiled agencies
step 2: saved agency tables and profiles
step 3: 66 agencies met the current review threshold
summary: 2258 evaluations, 222 agencies, 217 agencies with text-based profile outputs
```



## Streamlit Dashboard

The interactive dashboard allows leadership to review agency profiles, concern flags, and yearly trends without running the backend code.

```bash
streamlit run src/app.py
```

Run the pipeline at least once before launching the dashboard so the files in `outputs/` are available.



## Ethical Considerations

This project is framed as a decision-support and screening tool, not an automated judgment system.

Key commitments include:

* All outputs are aggregate-level and anonymized before any sharing
* Concern flags are intended to prompt leadership review, not penalize agencies
* TextBlob has limits when working with professional social work writing, so its results should be interpreted carefully
* The system is meant to be used alongside professional judgment and direct agency relationships, not instead of them
* Data access should remain limited to authorized Field Education personnel



## Limitations

* TextBlob performs better on emotionally expressive language than on professional social work writing, which is often more measured and descriptive
* Agencies with fewer than three responses are flagged as limited data and excluded from theme analysis
* The dataset reflects one program's placement history and may not generalize to other institutions without revising the theme dictionary and review logic
* Concern flags based on smaller response counts should be interpreted carefully



## Who This Is For

This project is designed for field education leadership teams, practicum coordinators, and program administrators at university social work programs who oversee student placements and collect end-of-placement evaluation data but do not yet have a structured analytics workflow for turning that feedback into clearer decision-support information.
