# Data Dictionary  
## Short-Form Video Consumption & Attention Span Study  

---

## How To Use This File

- Add one row for each column used in analysis or dashboarding.
- Explain what the field means in plain language.
- Mention any cleaning or standardization applied.
- Flag nullable columns, derived fields, and known quality issues.

---

## Dataset Summary

| Item | Details |
|---|---|
| Dataset name | Short-Form Video Consumption & Attention Span Study |
| Source | Synthetic behavioral dataset for DVA Capstone 2 — Newton School of Technology |
| Raw file name | raw_attention_study_data.csv |
| Clean file name | reels_attention_span_cleaned.csv |
| Last updated | April 29, 2026 |
| Granularity | One row per user (each record = one survey respondent) |
| Raw rows | 12,350 |
| Clean rows | 11,988 |
| Columns | 15 |

---

## Column Definitions

| Column Name | Data Type | Description | Example Value | Used In | Cleaning Notes |
|---|---|---|---|---|---|
| user_id | string | Unique identifier for each user | USR00042 | EDA | No nulls, used only as identifier |
| age | float | Age of user in years | 23.0 | EDA, KPI | Invalid values removed, nulls filled with median |
| gender | string | Self-reported gender | Female | EDA, Tableau | Standardised categories, null → Unknown |
| location | string | Residential area type | Urban | EDA | Trimmed whitespace, standardized |
| platform | string | Primary platform used | TikTok | KPI, Tableau | 15+ variants mapped to 3 platforms |
| reels_watch_time_hours | float | Daily reels consumption | 3.45 | KPI | Nulls filled, IQR clipping, logical constraints |
| daily_screen_time_hours | float | Total daily screen time | 8.5 | KPI | Converted to float, range filtered |
| scrolling_sessions_day | int | Sessions per day | 8 | EDA | Converted from float to int |
| notifications_per_day | int | Daily notifications | 28 | EDA | Nulls filled with median |
| sleep_hours | float | Average sleep hours | 7.2 | KPI | Invalid removed, median imputation |
| physical_activity_hours_week | float | Weekly physical activity | 4.5 | EDA | Highest missing column, median filled |
| stress_level | string | Stress category | Medium | KPI | Standardised to Low/Medium/High |
| attention_span_score | float | Attention rating (1–10) | 7.85 | KPI | Range enforced, outliers clipped |
| focus_level | float | Focus rating (1–10) | 5.5 | KPI | Non-numeric handled, median filled |
| task_completion_rate | float | % tasks completed | 42.5 | KPI | Range [0–100], nulls filled |

---

## Derived Columns

| Derived Column | Logic | Business Meaning |
|---|---|---|
| Cognitive Productivity Index (CPI) | (attention_span_score + focus_level + task_completion_rate/10) / 3 | Composite metric for cognitive performance |
| Consumption Tier | Based on reels_watch_time_hours bins | Segments users by consumption level |
| Risk Segment | IF reels>3 AND sleep<6 → Danger Zone, etc. | Identifies high-risk user groups |

---

## Data Quality Notes

- Missing values present in 6 columns — handled using median imputation  
- 350 duplicate rows removed during cleaning  
- Categorical inconsistencies fixed (platform, gender, stress)  
- Outliers handled using IQR clipping (not removal)  
- Logical constraints enforced:
  - reels_watch_time ≤ screen_time  
  - task_completion ∈ [0,100]  
  - attention & focus ∈ [1,10]  
- Synthetic dataset — relationships are structured, not purely organic  
- Self-reported fields may contain bias (attention, sleep, stress)  

---

## Notes for Reviewers

- Dataset is **cleaned, standardized, and analysis-ready**
- Supports:
  - EDA
  - KPI calculations
  - Tableau dashboards
- Designed for **behavioral + cognitive analysis**

---

## Related Files

- Cleaning Notebook: `01_Data_Cleaning.ipynb`  
- EDA Notebook: `02_EDA_Notebook.ipynb`  
- Dashboard: Tableau (.twbx)

---
