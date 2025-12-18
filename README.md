This project builds an end-to-end **customer churn prediction pipeline** for a telecommunications company.

The goal is to:
- Understand **which customers are likely to churn**.
- Identify **key drivers of churn** (usage, complaints, tariff plan, etc.).
- Provide **business recommendations** for targeted retention.

The work covers the full data science lifecycle: **EDA → preprocessing → class imbalance handling → feature engineering → feature selection → model comparison → final model & insights**.

---

## 1. Dataset

- File: `Customer_Churn.csv`
- Rows: 3,150 customers
- Target:
  - `Churn` (0 = no churn, 1 = churn)
- Main input features (simplified):
  - **Usage / behaviour**: `Seconds of Use`, `Frequency of use`, `Frequency of SMS`, `Distinct Called Numbers`
  - **Value**: `Customer Value`, `Charge Amount`
  - **Contract / service**: `Subscription Length`, `Status`, `Tariff Plan`, `Complains`, `Call Failure`
  - **Demographics**: `Age`, `Age Group`

The dataset is **imbalanced**: only ~15–16% of customers churn.

---

## 2. Methodology

### 2.1 Exploratory Data Analysis (EDA)

- Checked for **missing values** and **duplicates**.
- Used **descriptive statistics** (mean, median, std, variance, min, max).
- Analysed **shape** of distributions (skewness, kurtosis).
- Visualised:
  - Distributions of numeric features.
  - **Churn %** by `Complains`, `Status`, `Tariff Plan`, `Age Group`.
  - Boxplots of usage/value features split by `Churn`.

**Key EDA findings:**

- Churners are typically **low-activity, low-value** customers.
- Customers who **complain** and those with **Status = 2** churn much more.
- Tariff Plan 1 has higher churn than Tariff Plan 2.
- Usage features are **right-skewed** with some very heavy users (treated as genuine, not removed).

---

### 2.2 Preprocessing

- **Categorical encoding:**  
  Used `pd.get_dummies(..., drop_first=True)` for:
  - `Complains`, `Age Group`, `Tariff Plan`, `Status`  
  (plus engineered binary flags later).

- **Train–test split:**  
  `train_test_split` with:
  - `test_size=0.2`
  - `stratify=y` (to preserve churn proportion)

- **Feature scaling:**  
  `StandardScaler` applied to features for:
  - Logistic Regression  
  - SVM (RBF)  
  - MLP  
  - Naive Bayes  
  Random Forest used unscaled features (tree-based, scale-invariant).

---

### 2.3 Handling Class Imbalance

- Observed strong class imbalance (~84% non-churn, ~16% churn).
- Used **`RandomOverSampler`** (from `imblearn`) on the **training set**:
  - Oversampled `Churn = 1` until both classes were balanced.
  - Test set left unchanged to reflect real-world deployment.

Models were trained on:
1. **Imbalanced** data (baseline)
2. **Balanced** data (after oversampling)

---

### 2.4 Feature Engineering

To better capture customer behaviour and risk, three new features were created:

1. `Total_Activity`  
   - Sum of: `Seconds of Use + Frequency of use + Frequency of SMS + Distinct Called Numbers`  
   - Represents overall engagement.

2. `Low_Activity` (binary flag)  
   - `1` if `Total_Activity` is in the **bottom 25%** (Q1), else `0`.  
   - Low-activity customers have much higher churn rates.

3. `Low_Activity_and_Complain` (binary flag)  
   - `1` if `Low_Activity == 1` **and** `Complains == 1`, else `0`.  
   - This tiny segment is **extremely high churn** and business-critical.

Random Forest feature importance confirmed that:
- `Status_2`, `Total_Activity`, `Complains_1`, and `Customer Value` are among the **most important** predictors.

---

### 2.5 Feature Selection

Combined **statistical tests** and **model-based importance**:

- **Mann–Whitney U** tests for numeric features vs `Churn`:
  - Kept: `Total_Activity`, `Seconds of Use`, `Customer Value`, `Frequency of use`, `Frequency of SMS`, `Distinct Called Numbers`, `Charge Amount`.
  - Dropped: `Age`, `Call Failure` (no significant difference).
  - `Subscription Length` had borderline p-value but good Random Forest importance → kept.

- **Chi-square** tests for categorical features vs `Churn`:
  - Kept: `Complains`, `Age Group`, `Tariff Plan`, `Status`, `Low_Activity`, `Low_Activity_and_Complain`  
    (all strongly associated with churn).

Final feature set = **statistically significant + important** features.

---

### 2.6 Models

Trained and compared five classifiers at each stage:

- **Naive Bayes**
- **Logistic Regression**
- **SVM (RBF kernel)**
- **Random Forest Classifier**
- **MLP Neural Network**

**Metrics:**

- Accuracy  
- Precision (for churn class = 1)  
- Recall (for churn class = 1)  
- F1-score (for churn class = 1)  

---

## 3. Results (Summary)

### 3.1 Imbalanced vs Balanced

- On the **imbalanced** dataset:
  - Random Forest and MLP had strong accuracy and F1, but recall for churn was limited.
  - Naive Bayes had high recall but very low precision (many false alarms).

- On the **balanced** dataset (RandomOverSampler):
  - Accuracy stayed similar.
  - **Recall for churn increased**, especially for Random Forest and MLP.
  - Slight drop in precision, which is expected when detecting more churners.
  - F1-score for churn improved for the best models.

### 3.2 After Feature Engineering + Feature Selection

- Naive Bayes and Logistic Regression remained weaker on churn.
- SVM achieved very high recall but moderate precision.
- **Random Forest**: strong balance between precision and recall, good F1.
- **MLP**:  
  - Highest F1 for churn  
  - Very high recall (catches almost all churners)  
  → chosen as the **final model**.

---

## 4. Business Insights

Key insights derived from the models and EDA:

- **Low activity is a major churn signal**  
  Low_Activity customers, especially those who also complain, are at very high risk.

- **Complaints and service status matter**  
  Customers with `Complains = 1` and `Status = 2` churn much more than others.

- **Tariff Plan impact**  
  Tariff Plan 1 has higher churn than Plan 2, suggesting plan design or value issues.

- **Behavioural dataset**  
  The data mainly describes **what** customers are doing (usage patterns).  
  If available, combining it with data explaining **why** customers churn  
  (e.g. surveys, reasons for leaving, service issues) would improve the modelling and recommendations.

---


