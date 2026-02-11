# Weather Prediction Using LightGBM Machine Learning

## Project Description

This project aims to build a **weather prediction model** using the **LightGBM (LGBM)** machine learning algorithm based on historical weather parameter data.  
All analysis processes are implemented in a **Jupyter Notebook (`.ipynb`)**, covering data loading, data understanding, feature engineering, dataset splitting, model training, and evaluation.

This notebook demonstrates a complete **data preprocessing → modeling → evaluation** pipeline for a real-world weather prediction task.

---

## Analysis Workflow in the Notebook

### 1. Import Libraries
Main libraries used:
- `pandas` and `numpy` for data processing  
- `matplotlib` / `seaborn` for visualization  
- `scikit-learn` for preprocessing and evaluation  
- **`lightgbm`** as the primary prediction algorithm  

### 2. Load Dataset
The weather dataset is loaded into a dataframe for further processing.

### 3. Data Understanding
Exploratory analysis includes:
- Data structure and types  
- Descriptive statistics  
- Variable distributions  
- Missing value and anomaly detection  

### 4. Feature Engineering
Feature transformation to optimize **LightGBM** performance:
- Important feature selection  
- Categorical variable encoding  
- Normalization/standardization if required  

### 5. Train–Test Split
The dataset is divided into:
- **Training set**
- **Testing set**

This prevents *overfitting* and ensures objective evaluation.

### 6. Modeling with LightGBM
Core steps:
- Training a **LightGBM Classifier**
- Generating predictions on test data
- Analyzing model performance

### 7. Model Evaluation
Evaluation metrics include:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
## Prediction Result
![Prediction Result](prediction-result.jpeg)

---


