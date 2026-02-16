<div align="center">

# 🏡 King County Housing Market Analysis

*Predictive modeling of residential property values in King County, Washington*

**Ironhack Data Science Bootcamp - February 2026**

[View Presentation](./presentation/King_countyHousePrice_presentation.pdf) • [Explore Notebooks](./notebooks/)

---

</div>

## 🎯 Project Goal

Build machine learning models to accurately predict house prices in King County by analyzing property characteristics, location data, and market trends from 21,613 home sales.

## 📖 Table of Contents

- [About the Data](#-about-the-data)
- [Our Approach](#-our-approach)
- [What We Built](#-what-we-built)
- [Results](#-results)
- [Technologies](#-technologies)
- [Getting Started](#-getting-started)

---

## 🗂️ About the Data

**Source:** [Kaggle - King County House Sales](https://www.kaggle.com/datasets/minasameh55/king-country-houses-aa)

Our dataset contains **21,613 residential property transactions** with comprehensive information about each home:

### Property Features
- **Physical attributes**: bedrooms, bathrooms, square footage (living/lot/basement/above)
- **Quality indicators**: condition rating, grade, number of floors
- **Location data**: latitude, longitude, zipcode
- **Special features**: waterfront property, view quality
- **History**: year built, renovation year
- **Neighborhood context**: nearby properties' characteristics (15 neighbors)

### Data Characteristics
✅ Complete dataset - no missing values  
⚠️ Some properties appear multiple times (repeat sales)  
📊 Right-skewed price distribution  
🗺️ Clear geographic pricing patterns

---

## 🔬 Our Approach

### 1️⃣ **Exploratory Data Analysis**

We started by understanding the data landscape:

- Mapped geographic price distributions across King County
- Identified outliers in square footage and bedroom counts
- Discovered properties with questionable data (e.g., 0 bedrooms and 0 bathrooms)
- Analyzed price correlations with property features

**Key Discovery:** Location and living space dominate pricing, but luxury features create significant premiums.

### 2️⃣ **Data Preparation**

Cleaned and transformed the raw data:

```python
# Examples of our preprocessing
- Removed invalid entries (0 bedrooms and 0 bathrooms)
- Applied winsorization to handle extreme outliers
- Split sale dates into temporal features
```

### 3️⃣ **Feature Engineering**

Created meaningful features from raw data:

**Geographic Features**
- `loc_clusters` - Neighborhood groupings from coordinates
- `dist_seattle` - Proximity to downtown

**Property Metrics**
- `renovated` - Boolean: has the house been updated?
- `house_age` - Years since construction
- `total_living_ratio` - Indoor space relative to lot size
- `relative_size` - Comparison to neighboring properties

**Quality Indicators**
- `luxury_index` - Composite score: grade + view + 2×waterfront
- `quality_interaction` - Grade multiplied by condition
- `size_grade_interaction` - Large homes with high quality

**Sales History**
- `sold_occ` - Number of times property was sold

### 4️⃣ **Model Development**

Tested multiple algorithms with iterative improvement:

**Baseline → Linear Regression**  
Established performance floor

**Ensemble Methods → XGBoost, CatBoost, Gradient Boosting**  
Leveraged tree-based models for complex patterns

**Optimization → Hyperparameter Tuning**  
GridSearchCV + manual fine-tuning

---

## 🏗️ What We Built

### Model Portfolio

#### **Linear Regression (Baseline)**
Simple interpretable model for benchmarking
- R² = 0.703 (baseline) → 0.753 (with features)
- Validates feature engineering impact

#### **XGBoost Regressor** ⭐
Best overall performance with full feature set
- Test R² = **0.917**
- GridSearchCV → Optimal params: max_depth=7, learning_rate=0.1, n_estimators=300
- Experimented with dimensionality reduction (Top 10 features + PLS)

#### **CatBoost Regressor** ⭐
Top performer for categorical feature handling
- Test R² = **0.917** | MAE = **57,474** (best)
- Native support for categorical variables
- Depth=10, learning_rate=0.015, n_estimators=5000

#### **Gradient Boosting Regressor**
Systematic tuning approach
- RandomizedSearchCV → Manual optimization
- Reduced overfitting while maintaining R² = 0.905
- Best for understanding hyperparameter effects

---

## 📊 Results

### Performance Comparison

| Model | Test R² | RMSE | MAE | Highlight |
|-------|---------|------|-----|-----------|
| Linear (baseline) | 0.703 | 201,461 | 124,749 | Starting point |
| Linear (engineered) | 0.753 | 158,109 | 108,573 | +7% improvement |
| **XGBoost** | **0.917** | 91,876 | 58,869 | 🥇 Tied winner |
| **CatBoost** | **0.917** | 94,482 | **57,474** | Lowest MAE |
| GradientBoosting | 0.905 | 112,792 | 62,105 | Best generalization |

### What Matters Most?

**Top 5 Feature Importance (XGBoost)**
1. 🏠 **sqft_living** - Interior space is king
2. 🌊 **waterfront** - Massive price premium
3. 👁️ **view** - View quality drives value
4. ⭐ **grade** - Construction quality rating
5. 📏 **sqft_living15** - Neighborhood context

### Key Insights

💡 **Feature engineering delivered** - 7% R² boost for linear models  
💡 **Ensemble methods dominated** - 21% improvement over baseline  
💡 **Location matters enormously** - Geographic features critical  
💡 **CatBoost excels with mixed data** - Best for categorical handling  

---

## 🛠️ Technologies

```python
# Core Stack
Python 3.9
pandas, numpy          # Data manipulation
matplotlib, seaborn    # Visualization

# Machine Learning
scikit-learn          # Preprocessing, linear models, validation
xgboost              # Gradient boosting
catboost             # Categorical boosting
```

**Development Environment:** Jupyter Notebook  
**Data Source:** Kaggle API (`kagglehub`)

---

## 🚀 Getting Started

### Quick Start

```python
# 1. Download the data
import kagglehub
path = kagglehub.dataset_download("minasameh55/king-country-houses-aa")

# 2. Run the EDA notebook
jupyter notebook notebooks/king-county-house-prices-eda.ipynb

# 3. Experiment with models
jupyter notebook notebooks/Xgboost_regressor_experiments.ipynb
```

### Project Structure

```
king-county-house-prices/
│
├── data/
│   └── df_fe.csv          # processed dataset
│
├── notebooks/
│   ├── king-county-house-prices-eda.ipynb        # Full EDA
│   └── Xgboost_regressor_experiments.ipynb       # Model 
│
├── presentation/
│   └── King_countyHousePrice_presentation.pdf    # Final 
│
└── README.md
```
---

## 📜 License & Acknowledgments

**Data:** [King County House Sales dataset](https://www.kaggle.com/datasets/minasameh55/king-country-houses-aa) via Kaggle

This work was built together with [@alexandrade1978](https://github.com/alexandrade1978) and [@MariusGoeren](https://github.com/MariusGoeren) 🙌

---

<div align="center">


[⬆ Back to Top](#-king-county-housing-market-analysis)

</div>