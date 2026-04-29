# Section-A_G-2_ReelsAttentionSpan

> **Newton School of Technology | Data Visualization & Analytics**
> A 2-week industry simulation capstone using Python, GitHub, and Tableau to convert raw data into actionable business intelligence.

---

## Project Overview

| Field | Details |
|---|---|
| **Project Title** | Analysis of Reels Consumption Across Different Social Media Platforms|
| **Sector** | Digital Health & Consumer Technology |
| **Team ID** | G-2 |
| **Section** | A |
| **Faculty Mentor** | Aayushi Vashishth |
| **Institute** | Newton School of Technology |
| **Submission Date** | 29 April, 2026 |

### Team Members

| Role | Name | GitHub Username |
|---|---|---|
| Project Lead | Neelanshu Karn | karn-cyber |
| Data Lead |  Lakshya Agrawal | Lakshya182005 |
| ETL Lead | Aaryan Gera | whizkid-31 |
| Analysis Lead | Hrishi Dongre | hrishidongre |
| Visualization Lead | Akhilesh Kumar | UserAkku |
| Strategy Lead |  Kapil Karan Mathur | KapilKaranMathur |
| PPT and Quality Lead | Ritik Atri |  |
---

## Business Problem

Short-form video platforms — TikTok, Instagram Reels, and YouTube Shorts — have fundamentally reshaped digital consumption habits. Nearly half of surveyed users consume more than 3 hours of short-form content daily, against an average of only 6.5 hours of sleep per night — 1.5 hours below medical recommendations. This project investigates whether and how daily short-form content exposure correlates with measurable changes in attention span, focus level, and task completion across different demographic and behavioural segments, and identifies which user groups face the greatest cognitive risk.

**Core Business Question**

> How might we analyse short-form content consumption patterns to understand their impact on attention span across different user segments?

**Decision Supported**

> This analysis enables platform product teams, digital health regulators, and educators to identify at-risk user segments, establish evidence-based consumption thresholds, and design targeted digital wellness interventions.

---

## Dataset

| Attribute | Details |
|---|---|
| **Source Name** | reels_attention_span_dirty.csv |
| **Direct Access Link** | [Data Link](https://www.kaggle.com/datasets/jayjoshi37/reelsshorts-consumption-vs-attention-span) |
| **Row Count** | 8,926 records |
| **Column Count** | 10 columns |
| **Time Period Covered** | Cross-sectional survey (single wave) |
| **Format** | CSV |

**Key Columns Used**

| Column Name | Description | Role in Analysis |
|---|---|---|
| `age` | User age in years (15–44) | Segmentation — age group creation, vulnerability analysis |
| `reels_watch_time_hours` | Daily hours spent watching short-form content | Primary behavioral input; consumption tier classification; KPI |
| `daily_screen_time_hours` | Total daily screen time in hours | Context variable; reels-to-screen ratio computation |
| `sleep_hours` | Average nightly sleep hours | Risk factor; Danger Zone segmentation |
| `attention_span_score` | Self-rated attention ability on a 1–10 scale | Primary cognitive outcome KPI |
| `focus_level` | Self-rated focus ability on a 1–10 scale | Secondary cognitive outcome; CPI component |
| `task_completion_rate` | Percentage of daily tasks completed (0–100) | Tertiary cognitive outcome; CPI component |
| `stress_level` | Perceived stress tier — Low / Medium / High | Segmentation and comparative analysis |
| `platform` | Primary short-form video platform used | Grouping and platform-level comparison |

For full column definitions, see [`docs/data_dictionary.md`](docs/data_dictionary.md).

---

## KPI Framework

| KPI | Definition | Formula / Computation |
|---|---|---|
| **Cognitive Performance Index (CPI)** | A single composite score normalising all three cognitive outcome variables onto a 0–10 scale; the primary outcome KPI used throughout all dashboards | `(attention_span_score/10 + focus_level/10 + task_completion_rate/100) / 3 × 10` — computed in `02_EDA___Analysis.ipynb` Feature Engineering section |
| **% Over-Consuming Users** | Proportion of users exceeding the 3-hour daily reels threshold, defined as the boundary between moderate and heavy use | `COUNT(reels_watch_time_hours > 3) / COUNT(all users) × 100` |
| **Reels Share of Screen Time (%)** | Share of total daily screen time dedicated to short-form content; used for platform-level benchmarking | `reels_watch_time_hours / daily_screen_time_hours × 100` — stored as `reels_ratio_pct` in the cleaned dataset |
| **Danger Zone User %** | Proportion of users simultaneously exceeding the reels threshold AND sleeping under 6 hours — the compound-risk population | `COUNT(reels > 3h AND sleep < 6h) / COUNT(all users) × 100` |
| **Risk Segment** | Four-way behavioural classification enabling targeted intervention design | `IF reels > 3 AND sleep < 6 → "Danger Zone"` / `IF reels > 3 ONLY → "Reels at Risk"` / `IF sleep < 6 ONLY → "Sleep Deprived"` / `ELSE → "Safe Zone"` — computed via `segment()` function in `02_EDA___Analysis.ipynb` |
| **Age Group** | Five-band demographic segmentation used across all comparative analyses | `pd.cut(age, bins=[14,18,24,30,36,44], labels=["15-18","19-24","25-30","31-36","37-44"])` — computed in Feature Engineering |
| **Reels Consumption Tier (reels_bin)** | Five-level ordinal classification of daily reels consumption | `< 1h / 1–2h / 2–3h / 3–4h / 4–6h` — computed via `pd.cut` on `reels_watch_time_hours` |

Document KPI logic clearly in `notebooks/04_statistical_analysis.ipynb` and `notebooks/05_final_load_prep.ipynb`.

---

## Tableau Dashboard

| Item | Details |
|---|---|
| **Dashboard URL** | _Paste Tableau Public link here_ |
| **Executive View** | Dashboard 1 (Consumption Exposure) + Dashboard 2 (Cognitive Health Overview) — population-level KPI cards, consumption distribution, and age-group attention benchmarks for senior stakeholders |
| **Operational View** | Dashboard 3 (Risk Segmentation) + Dashboard 4 (Intervention Threshold) — four-segment risk matrix, Age × Reels heatmap, CPI by consumption tier, and evidence-based threshold analysis for product and policy teams |
| **Main Filters** | Platform (TikTok / Instagram Reels / YouTube Shorts), Age Group (15–18 / 19–24 / 25–30 / 31–36 / 37–44), Stress Level (Low / Medium / High) — all filters apply across all worksheets |

Store dashboard screenshots in [`tableau/screenshots/`](tableau/screenshots/) and document the public links in [`tableau/dashboard_links.md`](tableau/dashboard_links.md).

---

## Key Insights

1. **No single variable predicts cognitive decline.** All Pearson correlations between consumption/sleep variables and attention/focus/CPI fall below |r| = 0.03. Cognitive performance is multi-factorial — single-variable interventions will be insufficient.
2. **Young adults (19–24) are the most vulnerable segment.** They have the lowest mean attention score (5.38) of any age group and the highest average daily reels consumption (3.1 h/day) — both conditions are simultaneously present.
3. **19.3% of users are in the Danger Zone.** Approximately 1,723 users simultaneously exceed 3 h of daily reels AND sleep under 6 hours per night — the clearest and most urgent intervention target.
4. **Platform choice does not affect cognitive outcomes.** TikTok, Instagram Reels, and YouTube Shorts produce statistically indistinguishable results across all cognitive and behavioural variables. The problem is consumption volume, not the specific platform.
5. **Age moderates the consumption-cognition relationship.** The Age × Reels heatmap shows 19–24 users at moderate consumption (2–4 h) scoring lower (5.27–5.33) than 37–44 users at high consumption (4–6 h), who remain above 5.65. Age acts as a protective factor in older adults.
6. **48.2% of users are over-consuming.** Nearly half the sample watches 3+ hours of reels daily. A 2-hour daily cap — the lower bound of the safe range — would require behaviour change from more than 50% of the population.
7. **40.2% of users sleep under 6 hours nightly.** Well below the medically recommended 8 hours, and co-occurring heavily with heavy reels use — making sleep deprivation both an independent risk and a compounding factor.
8. **Cognitive score distributions are uniform, not normal.** Attention, focus, and task completion are spread nearly flat across their full ranges (kurtosis ≈ −1.19). This high individual variability explains why group means hover near the midpoint (~5.5) regardless of consumption levels.
9. **Stress level has no significant effect on cognitive outcomes.** ANOVA across Low/Medium/High stress groups shows p > 0.05 for all cognitive outcome variables. Mean attention scores are 5.44 (Low), 5.54 (Medium), 5.54 (High) — within noise.
10. **The dataset captures adaptation, not damage.** Cross-sectional near-zero correlations do not mean short-form content is harmless. Users have adapted to their habits. Longitudinal studies consistently show attention degradation over months of heavy use — a critical limitation to disclose.

---

## Recommendations

| # | Insight Addressed | Recommendation | Expected Impact |
|---|---|---|---|
| 1 | Insight 6 — 48.2% over-consuming | Implement a 2–3 hour daily reels usage cap with progressive friction (session summaries at 2h, configurable hard limit at 3h) built directly into platform UX | Estimated reduction of Danger Zone population from 19.3% to under 10% at 60% user adherence |
| 2 | Insight 2 — 19–24 most vulnerable | Deploy age-segmented digital wellness programmes targeting the 19–24 cohort via university partnerships, in-app awareness modules, and attention-span training tools | Potential 0.15–0.25 point improvement in mean attention score for the 19–24 segment |
| 3 | Insight 7 — 40.2% sleep-deprived | Integrate sleep-awareness nudges into platform UX: opt-in bedtime reminders, automatic night-mode after 10 PM, and monthly sleep-linked usage reports | Reduction of Sleep Deprived segment from 20.8% toward a realistic population norm of under 15% |
| 4 | Insight 10 — cross-sectional limitation | Commission a 12-month longitudinal panel study tracking attention scores before and after controlled consumption habit changes to establish causal evidence for regulation | Transition from associative to causal evidence base; enables regulatory-grade policy advocacy |
| 5 | Insight 1 — multi-factorial outcomes | Adopt the Cognitive Performance Index (CPI) as an industry-standard wellness benchmark, reported quarterly by platforms as part of digital wellbeing transparency disclosures | Provides a shared measurement language across regulators, researchers, and product teams for cross-platform comparison |

---

## Repository Structure

```text
SectionName_TeamID_ReelsAttentionSpan/
|
|-- README.md
|
|-- data/
|   |-- raw/
|   |   `-- reels_attention_span_dirty.csv     
|   `-- processed/
|       `-- cleaned_data.csv                    
|-- notebooks/
|   |-- 01_Extraction_&_Cleaning.ipynb          
|   |-- 02_EDA_&__Analysis.ipynb                  
|                 
|
|-- scripts/
|   `-- etl_pipeline.py
|
|-- tableau/
|   |-- screenshots/
|   `-- dashboard_links.md
|
|-- reports/
|   |-- project_report.pdf
|   |-- presentation.pdf
|   
|
|-- docs/
|   `-- data_dictionary.md
|
|-- DVA-oriented-Resume/
`-- DVA-focused-Portfolio/
```

---

## Analytical Pipeline

The project follows a structured 7-step workflow:

1. **Define** — Research question scoped around short-form content consumption and attention span impact across user segments; mentor approval obtained.
2. **Extract** — Raw dataset (`reels_attention_span_dirty.csv`, 8,926 records, 10 columns) sourced and committed to `data/raw/`; data dictionary drafted in `docs/data_dictionary.md`.
3. **Clean and Transform** — Full cleaning pipeline built in `notebooks/01_Extraction___Cleaning.ipynb`: numeric type coercion, platform label normalisation (e.g. "IG" / "insta" → "Instagram Reels"), domain rule enforcement, median imputation, IQR-based clipping, duplicate removal, and logical consistency checks. Output saved to `data/processed/cleaned_data.csv`.
4. **Analyze** — EDA performed in `notebooks/02_EDA___Analysis.ipynb` across three levels: univariate (distributions, skewness, kurtosis), bivariate (Pearson correlations, ANOVA, group comparisons), and multivariate (heatmaps, risk segmentation, ECDF, pair plots). Feature engineering produced `age_group`, `reels_bin`, `Segment`, `CPI`, and `reels_ratio_pct`.
5. **Visualize** — Four interactive Tableau dashboards built covering Consumption Exposure, Cognitive Health Overview, Risk Segmentation, and Intervention Threshold — published on Tableau Public.
6. **Recommend** — Five data-backed business recommendations delivered, each directly linked to a numbered insight from the analysis.
7. **Report** — Final project report (`reports/project_report.pdf`) and presentation deck completed and exported to PDF.

---

## Tech Stack

| Tool | Status | Purpose |
|---|---|---|
| Python  | Mandatory | ETL, cleaning, analysis, and KPI computation |
| Google Colab | Supported | Cloud notebook execution environment |
| Tableau Public | Mandatory | Dashboard design, publishing, and sharing |
| GitHub | Mandatory | Version control, collaboration, contribution audit |


**Python libraries used:** `pandas`, `numpy`, `matplotlib`, `seaborn`

---


## Contribution Matrix

This table must match evidence in GitHub Insights, PR history, and committed files.

| Team Member | Dataset & Sourcing | ETL & Cleaning | EDA & Analysis | Statistical Analysis | Tableau Dashboard | Report Writing | PPT & Viva |
|---|---|---|---|---|---|---|---|
| Neelanshu Karn |  |  | ✓ | ✓ | | | ✓|
| Hrishi Dongre |  ✓ |  | ✓ |  | ✓ |  |  |
| Akhilesh Kumar | ✓ |  |  |  | ✓ | ✓ |  |
| Lakshya Agrawal | ✓ |  |  | ✓ | ✓ |  |  |
| Kapil Kumar Mathur |  |  | ✓ | ✓ |  | ✓ |  |
| Ritik Atri | ✓ |  |  |  |  | ✓ | ✓ |
| Aaryan Gera |  | ✓ | ✓ | ✓ |  |  |  |

_Declaration: We confirm that the above contribution details are accurate and verifiable through GitHub Insights, PR history, and submitted artifacts._

**Team Lead Name:** Neelanshu Karn

**Date:** 29 April, 2026

---

## Academic Integrity

All analysis, code, and recommendations in this repository must be the original work of the team listed above. Free-riding is tracked via GitHub Insights and pull request history. Any mismatch between the contribution matrix and actual commit history may result in individual grade adjustments.

---

*Newton School of Technology — Data Visualization & Analytics | Capstone 2*