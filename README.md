# 📈 Sales Prediction Engine

This project is a **Sales Prediction Engine** built using machine learning techniques on an advertising dataset. The model predicts product sales based on advertising spend across **TV**, **Radio**, and **Newspaper** channels.

---

## 📂 Dataset

- **Source**: Local CSV file (`advertising.csv`)
- **Rows**: 201
- **Columns**:
  - `TV`: Advertising budget spent on TV (in thousands of dollars)
  - `Radio`: Advertising budget spent on Radio
  - `Newspaper`: Advertising budget spent on Newspapers
  - `Sales`: Units sold (target variable)

---

## 🧠 Objective

To predict `Sales` using advertising budgets and identify which marketing channel contributes most to sales performance.

---

## ⚙️ Tech Stack

- **Python**
- **Pandas, NumPy**
- **Seaborn, Matplotlib**
- **Scikit-learn**
- **XGBoost**
- **Jupyter Notebook**

---

## 🔍 Workflow

### 1. Exploratory Data Analysis (EDA)
- Visualized relationships using pair plots and correlation heatmaps.

### 2. Data Cleaning
- Checked for missing values and outliers.
- Ensured correct data types.

### 3. Feature Engineering
- Created interaction term: `TV * Radio`.

### 4. Preprocessing
- Standardized features for linear models.
- Split into train/test sets (80/20).

### 5. Model Building
Tested multiple regression models:
- ✅ Linear Regression
- ✅ Random Forest Regressor
- ✅ XGBoost Regressor

### 6. Evaluation Metrics
- **RMSE** (Root Mean Squared Error)
- **R² Score**

### 7. Best Model
- ✅ **XGBoost Regressor**
  - RMSE: `1.01`
  - R² Score: `0.97`

---

## 📊 Model Performance

| Model            | RMSE | R² Score |
|------------------|------|----------|
| Linear Regression| 1.54 | 0.92     |
| Random Forest    | 1.09 | 0.96     |
| **XGBoost**      | 1.01 | 0.97     |

---

## 🧪 Example Prediction

```python
new_data = pd.DataFrame({
    'TV': [100],
    'Radio': [20],
    'Newspaper': [10],
})
new_data['TV_Radio'] = new_data['TV'] * new_data['Radio']
prediction = xgb.predict(new_data)
print("Predicted Sales:", prediction[0])
