# CIE 2 – Activity 2: Binary Classification using Logistic Regression

**Institution:** Zeal Education Society's Zeal College of Engineering and Research (ZCOER), Pune

**Department:** Artificial Intelligence & Data Science (AI & DS)

**Class:** T.Y. B.Tech | **Semester:** V | **Division:** `[Your Division]`

**Roll Number:** `[Your Roll Number]` | **Name:** `[Your Name]`

**Course:** Regression and Classification Algorithms (Unit II – Classification & Logistic Regression)

**Academic Year:** 2026–27

**Submission GitHub ID:** <https://github.com/complexiota>

---

## 1. Activity Overview & Problem Statement

This repository contains the complete implementation and analysis for **Activity 2: Binary Classification Activity**. Acting in the role of a data scientist at an academic institution, the objective is to build a **Logistic Regression** model that predicts whether a student will **Pass (`1`)** or **Fail (`0`)** based on academic and lifestyle indicators.

The model outputs probabilities rather than just hard labels, allowing academic advisors to evaluate error trade-offs, compare decision thresholds (`0.3`, `0.5`, and `0.7`), and identify at-risk students early for mentoring or remedial support.

---

## 2. Dataset Description

* **Dataset Name:** Student Performance Dataset (`student_performance.csv`)
* **Source:** Synthetically generated educational dataset created for this academic practical (non-personal, non-confidential data).
* **Size:** 1,000 rows $\times$ 7 columns (6 input features + 1 binary target).
* **Data Quality:** 0 missing values, 0 duplicate rows.
* **Class Balance:** `597` Pass (`59.7%`) and `403` Fail (`40.3%`) — moderately balanced.
* **Stratified 80:20 Split (`random_state=42`):**
  * **Training Set:** 800 rows (`478` Pass, `322` Fail)
  * **Testing Set:** 200 rows (`119` Pass, `81` Fail)

### Feature Dictionary & Class Definitions

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `Attendance_Pct` | Numerical | Student attendance percentage |
| `Study_Hours` | Numerical | Average daily study hours |
| `Assignment_Score` | Numerical | Score obtained in coursework/assignments |
| `Previous_Exam_Score` | Numerical | Score from the previous examination |
| `Sleep_Hours` | Numerical | Average daily sleep duration (hours) |
| `Internet_Access` | Categorical | Access to internet for study (`Yes` / `No`, encoded as `1` / `0`) |
| `Pass` **(Target)** | Binary | **`1` = Pass** (Positive Class), **`0` = Fail** (Negative Class) |

---

## 3. Work Completed

The Google Colab notebook (`Activity2_Logistic_Regression_Student_Performance.ipynb`) is structured into **11 concise cells** covering every required step:

1. **Data Loading & Inspection:** Checked shape `(1000, 7)`, data types, missing values (`0`), duplicates (`0`), and plotted the class distribution bar chart.
2. **Feature Preparation & Stratified Split:** Binary-encoded `Internet_Access`, performed an **80:20 stratified train-test split** (`random_state=42`), and applied `StandardScaler` fit strictly on the training set to prevent data leakage.
3. **Sigmoid Function:** Plotted the sigmoid curve $\sigma(z) = \frac{1}{1 + e^{-z}}$ illustrating how linear scores map to probabilities between $0$ and $1$ with a $0.5$ decision threshold.
4. **Model Training & Coefficients:** Trained `LogisticRegression` on the scaled training set and extracted the intercept and feature coefficients to interpret positive/negative influence on passing probability.
5. **Predicted Probabilities & Labels:** Generated test-set probabilities and class predictions, displaying a side-by-side sample with actual labels.
6. **Confusion Matrix:** Visualised the labelled confusion matrix and interpreted `TP`, `TN`, `FP`, and `FN` in the student-advising context.
7. **Evaluation Metrics & Baseline Comparison:** Calculated Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC, full classification report, majority-class baseline, and train vs. test accuracy check.
8. **ROC Curve:** Plotted the Receiver Operating Characteristic (ROC) curve and computed the ROC-AUC score.
9. **Threshold Study:** Evaluated model behaviour across decision thresholds `0.3`, `0.5`, and `0.7` to examine the Precision–Recall trade-off.
10. **Decision Behaviour Visualisation:** Plotted predicted Pass probability against `Previous_Exam_Score` with the $0.5$ threshold line.
11. **New Student Prediction:** Applied the exact preprocessing pipeline to predict the probability and outcome for a new student record.

---

## 4. Key Results

### Learned Model Parameters (Standardised Features)

* **Intercept:** `0.6802`

| Feature | Coefficient | Influence on Passing Probability |
| :--- | :---: | :--- |
| `Study_Hours` | `+1.1343` | Strongest positive predictor |
| `Previous_Exam_Score` | `+1.0934` | Strong positive predictor |
| `Attendance_Pct` | `+0.9265` | Strong positive predictor |
| `Assignment_Score` | `+0.8011` | Moderate-to-strong positive predictor |
| `Internet_Access` | `+0.3992` | Moderate positive predictor |
| `Sleep_Hours` | `+0.1783` | Mild positive predictor |

*Interpretation:* All six features have positive coefficients, meaning higher values increase the log-odds (and therefore probability) of a student passing, with `Study_Hours` and `Previous_Exam_Score` having the largest relative impact.

### Baseline & Generalisation Check

* **Majority-Class Baseline Accuracy (Test Set):** `0.5950` (`59.50%`, always predicting `Pass`)
* **Training Set Accuracy:** `0.7775` (`77.75%`)
* **Testing Set Accuracy:** `0.8250` (`82.50%`)
* **Observation:** The model outperforms the majority-class baseline by `+23.00%` on unseen test data and shows no signs of overfitting.

### Confusion Matrix Summary (Test Set, Threshold = 0.5)

* **True Positives ($TP$):** `110` — Correctly predicted Pass
* **True Negatives ($TN$):** `55` — Correctly predicted Fail
* **False Positives ($FP$):** `26` — Predicted Pass, actually Failed *(most costly error: at-risk student misses remedial help)*
* **False Negatives ($FN$):** `9` — Predicted Fail, actually Passed *(low-cost error: passing student offered optional extra help)*

### Classification Evaluation Results (Test Set, Threshold = 0.5)

| Metric | Formula | What It Tells the User | Test-Set Result |
| :--- | :--- | :--- | :---: |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Share of all predictions that are correct | `0.8250` |
| **Precision** | $\frac{TP}{TP + FP}$ | Of students predicted to Pass, how many actually Passed | `0.8088` |
| **Recall (Sensitivity)** | $\frac{TP}{TP + FN}$ | Of students who actually Passed, how many the model identified | `0.9244` |
| **Specificity** | $\frac{TN}{TN + FP}$ | Of students who actually Failed, how many the model correctly flagged | `0.6790` |
| **F1-Score** | $\frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ | Single score balancing Precision and Recall | `0.8627` |
| **ROC-AUC** | Area under ROC curve | Ability to rank Pass cases above Fail cases across all thresholds | `0.8954` |

*Interpretation:* With an ROC-AUC of `0.8954` and Accuracy of `82.50%`, the model reliably ranks passing students above failing students well above random guessing (`0.50`) and the majority-class baseline (`59.50%`).

### Threshold Comparison Table

| Threshold | Predicted Positive | Accuracy | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.3** | `161` | `0.7500` | `0.7143` | `0.9664` | `0.8214` |
| **0.5** | `136` | `0.8250` | `0.8088` | `0.9244` | `0.8627` |
| **0.7** | `100` | `0.7950` | `0.8900` | `0.7479` | `0.8128` |

*Trade-off Observation:* Raising the threshold from `0.3` to `0.7` reduces the number of students predicted to Pass (`161` $\rightarrow$ `100`), increasing **Precision** (`0.7143` $\rightarrow$ `0.8900`) while decreasing **Recall** (`0.9664` $\rightarrow$ `0.7479`).

### New Student Prediction (Cell 11)

* **Input Profile:** `Attendance_Pct = 82.0`, `Study_Hours = 4.0`, `Assignment_Score = 75.0`, `Previous_Exam_Score = 70.0`, `Sleep_Hours = 7.0`, `Internet_Access = Yes (1)`
* **Predicted Probability of Passing:** `0.9450` (`94.50%`)
* **Predicted Class:** `Pass (1)`

---

## 5. Short Decision Log

* **Problem Framing & Error Cost:** In student advising, a **False Positive (FP)**—predicting a struggling student will pass when they actually fail—is more costly than a **False Negative (FN)**, because the student misses timely remedial support.
* **Key Metrics Prioritised:** **Precision** (minimising False Positives), **Specificity** (correctly catching failing students), and **ROC-AUC** (overall ranking quality) were prioritised alongside F1-Score.
* **Threshold Recommendation:**
  * Use **0.5** for the best overall balance between Precision and Recall (`F1 = 0.8627`, `Accuracy = 82.50%`).
  * Use **0.7** if the institution wants a stricter standard for marking a student as "safe to pass" (raises Precision to `89.00%` and catches more at-risk students for early intervention, at the cost of flagging more passing students for optional review).
* **Limitations & Ethics:** Predictions are probability-based estimates, not guaranteed outcomes. This model must **never** be used alone to make punitive academic decisions (such as barring a student from exams); it should only assist faculty in offering proactive support.

---

## 6. Repository Structure & How to Run

```text
├── Activity2_Logistic_Regression_Student_Performance.ipynb   # Main Google Colab notebook (11 cells)
├── student_performance.csv                                   # Synthetic dataset (1,000 records)
└── README.md                                                 # Project documentation & results
```

### Steps to Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/) and upload `Activity2_Logistic_Regression_Student_Performance.ipynb`.
2. Open the **Files** sidebar and upload `student_performance.csv`.
3. Select **Runtime → Run all** to reproduce all tables, metrics, and plots.