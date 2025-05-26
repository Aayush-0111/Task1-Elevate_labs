# Task1-Elevate_labs
# 🛳️ Titanic Dataset Preprocessing

This repository contains a complete preprocessing pipeline for the Titanic dataset using Python, `pandas`, `scikit-learn`, and `seaborn`. The goal is to clean, transform, and visualize the dataset to prepare it for further analysis or machine learning.

---

## 📂 Dataset

- **File**: `Titanic-Dataset.csv`
- **Source**: Common Titanic dataset used for data science practice
- **Features Included**: `Age`, `Fare`, `Sex`, `Embarked`, `Pclass`, `Survived`, and more

---

## 🔧 Preprocessing Steps

### 1. 📥 Import and Explore Data
- Load dataset using `pandas`
- Check basic info: shape, nulls, data types

### 2. 🧹 Handle Missing Values
- `Age`: Filled with **mean** using `SimpleImputer`
- `Embarked`: Filled with **mode**
- `Cabin`: Dropped due to excessive missing data

### 3. 🔢 Encode Categorical Variables
- Used `LabelEncoder` to convert:
  - `Sex`: male/female → 0/1
  - `Embarked`: C/Q/S → 0/1/2

### 4. ⚖️ Normalize Numerical Features
- Applied **StandardScaler** to:
  - `Age`
  - `Fare`

### 5. 📊 Visualize & Remove Outliers
- Boxplots for `Age` and `Fare`
- Outliers removed using the **IQR method**

### 6. 🧪 Additional Visualizations
- Boxplots grouped by:
  - `Pclass` vs `Age`
  - `Survived` vs `Age`
- Enhanced with notches and color styling

---

## 📦 Dependencies

Install with pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
