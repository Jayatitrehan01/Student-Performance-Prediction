# Student Performance Prediction using Machine Learning

# Dataset Description
The dataset used in this project comes from the UCI Machine Learning Repository and focuses on student achievement in secondary education (high school) in Portugal. It contains detailed information on students enrolled in two subjects: Mathematics and Portuguese Language.

For this project, we used the student-mat.csv file, which includes 395 records of students taking the Math course.

---
# Feature Description
* Personal- age, sex, address, famsize, Pstatus
* Academic- studytime, failures, schoolsup, paid, G1, G2, G3
* Social- freetime, goout, Dalc, Walc, romantic, internet, nursery
* Family- Medu, Fedu, Mjob, Fjob, guardian, famrel
* Target	G3 — final math grade (our prediction target)
---

# Project Steps

### 1️⃣ Data Loading & Initial Exploration
- Loaded `student-mat.csv` using Pandas.
- Checked shape, types, and missing values (`no nulls`).
- Renamed columns for readability (e.g., `sex → gender`, `G3 → final_grade`).
- Converted `object` types to `category`.

### 2️⃣ Exploratory Data Analysis (EDA)
- Plotted:
  - Final grade distribution
  - Study time histogram
  - Absences vs Grade scatter plot
  - KDE curve for G3
- Created a correlation heatmap
- Grouped and analysed final grade averages across:
  - Gender, parental jobs, school support, paid classes, etc.

### 3️⃣ Feature Engineering
- Separated features:
  - `cat_cols`: Categorical (e.g., `mother_job`)
  - `num_cols`: Numeric (e.g., `age`, `studytime`)
  - `outliers`: Features needing log transformation (`absences`)
  - `ordinal_enc`: Ordinal categorical values (`Medu`, `Fedu`, `higher`)
- Applied:
  - `PowerTransformer` for numeric features
  - `FunctionTransformer (np.log1p)` for outliers
  - `OneHotEncoder` for nominal categorical
  - `OrdinalEncoder` for ordinal features
- Combined preprocessing using `ColumnTransformer` + `Pipeline`

### 4️⃣ Data Split & Preprocessing
- Split data into train and test sets (`80/20` split).
- Transformed features using preprocessing pipeline:
  - `X_train_transformed`, `X_test_transformed`

---

## 🤖 Model Training

### Models Trained:
| Model                      | Hyperparameter Tuning |
|---------------------------|------------------------|
| Linear Regression          | ❌                    |
| Lasso Regression (LassoCV) | ✅ (`alpha` tuning)     |
| Decision Tree Regressor   | ✅ (`max_depth`, `min_samples`) |
| Random Forest Regressor   | ✅ (`n_estimators`, `max_features`, etc.) |
| Gradient Boosting Regressor | ✅ (`learning_rate`, `n_estimators`) |
| Support Vector Regressor (SVR) | ✅ (`C`, `gamma`, `epsilon`) |

Used `RandomizedSearchCV` for tree-based models and SVR with 5-fold CV and R² scoring.

---

## Model Evaluation

- **Metrics**:  
  - R² Score (goodness of fit)  
  - RMSE (Root Mean Squared Error)  
  - 10-fold Cross-Validation on training set

- **Best Performing Model**: ✅ **Gradient Boosting Regressor**
  - R² Score: **0.852**
  - RMSE: **1.74**
  - CV R²: **0.8619**

### 🔢 Final Comparison Table

| Model                       | R² Score | RMSE  |
|-----------------------------|----------|--------|
| Gradient Boosting Regressor | 0.852    | 1.74   |
| Random Forest Regressor     | 0.836    | 1.83   |
| Lasso Regression            | 0.799    | 2.02   |
| Linear Regression           | 0.782    | 2.11   |
| Decision Tree Regression    | 0.770    | 2.17   |
| SVR                         | 0.739    | 2.31   |

---

## 📊 Visualizations

- ✅ **RMSE and R² Score Comparison** (Bar Plots)
- ✅ **Actual vs Predicted Grades** (Scatter Plot)
- ✅ **Boxplots** for detecting outliers
- ✅ **Correlation Heatmap**
- ✅ **KDE and Histograms**

---

