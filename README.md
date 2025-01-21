# Project: Credit Card Approval Prediction

This project leverages data mining and machine learning techniques to predict credit card approval. It uses various classifiers, including Logistic Regression, Decision Tree, and Random Forest, to analyze and predict outcomes based on applicant data.

---

## Overview

The objective of this project is to design a robust system that predicts whether a credit card application will be approved or not, based on key features of the applicant. The project includes detailed steps for data preprocessing, feature scaling, model training, evaluation, and hyperparameter tuning. The ultimate outcome is a robust and adaptive system that ensures fairness, accuracy, and scalability.

---

## Features

- **Data Preprocessing**: Handling missing values, encoding categorical variables, and feature scaling.
- **Model Implementation**: Logistic Regression, Decision Tree, and Random Forest classifiers.
- **Hyperparameter Tuning**: Using GridSearchCV for optimal parameter selection.
- **Model Evaluation**: Confusion matrix, classification report, and ROC curve analysis.

---

## Dataset Information

The dataset used for this project was sourced from Kaggle and contains both numerical and categorical variables relevant to credit card approval decisions. Key characteristics of the dataset include:
- **Demographic Features**: Age, Gender, Marital Status.
- **Financial Attributes**: Annual Income, Debt Levels, Credit Score.
- **Miscellaneous Features**: Employment History, Property Ownership, Housing Type.

### Preprocessing Steps:
1. **Handling Missing Data**: Missing values were imputed.
2. **Outlier Removal**: The Interquartile Range (IQR) method was applied to address extreme values in features like `Annual Income` and `Children`.
3. **Feature Scaling**: Numerical variables were normalized to ensure consistency.
4. **Encoding Categorical Features**: Variables such as Gender and Marital Status were converted to machine-readable formats using one-hot encoding.
5. **Feature Engineering**: New features, such as the credit utilization ratio and age categories, were derived to enhance predictive power.
6. **Class Balancing**: Oversampling techniques were used to address class imbalance between approved and rejected applications.

---

## Methodology

The project followed a structured pipeline:

### 1. **Exploratory Data Analysis (EDA)**
- Examined distributions of features such as Gender, Car Ownership, and Annual Income using visualizations like bar charts and histograms.
- Correlation heatmaps were used to identify relationships between features (e.g., Income vs. Approval Rates).
- Outliers were identified and removed to improve data quality.

### 2. **Feature Engineering**
- Derived features like `Age Categories` and `Employment Duration` to enhance model interpretability.
- Encoded categorical variables to enable machine learning compatibility.

### 3. **Model Training and Evaluation**

Three classifiers were implemented and evaluated:

1. **Logistic Regression**:
   - A simple and interpretable linear model.
   - Evaluated using metrics such as accuracy, confusion matrix, and ROC-AUC.

2. **Decision Tree**:
   - Captures non-linear patterns and interactions between features.
   - Evaluated on metrics like accuracy, tree depth, and feature importance.

3. **Random Forest**:
   - An ensemble method that combines multiple decision trees for higher accuracy and robustness.
   - Metrics include accuracy, confusion matrix, and ROC-AUC.

### Hyperparameter Tuning

- **GridSearchCV**: Used to fine-tune hyperparameters for all classifiers, ensuring the best performance.

---

## Results

### Model Evaluation and Performance
The table below summarizes the performance of the implemented models:

| Model Name           | Accuracy | Precision | F1-Score | ROC-AUC | Mean Cross-Validation Accuracy |
|----------------------|----------|-----------|----------|---------|--------------------------------|
| Random Forest (RFC)  | 90%      | 0.92      | 0.95     | 0.74    | 90%                            |
| Logistic Regression (LRC) | 87%      | 0.88      | 0.89     | 0.61    | 87%                            |
| Decision Tree (DT)   | 85%      | 0.86      | 0.87     | 0.71    | 86%                            |

### Feature Importance

![image](https://github.com/user-attachments/assets/523c22f8-42e6-4c05-a2a7-4c19f0141f95)

The feature importance plot for the Random Forest model highlights the most significant variables:

- **Employment Years** and **Age** are the most influential features, indicating that stability in employment and the applicant's age significantly impact credit approval decisions.
- **Annual Income** also contributes strongly, showcasing the importance of financial capacity.
- **Type of Income** and **Marital Status** provide additional insights into financial stability and dependability.
- **Education** has a relatively lower impact compared to other features but still plays a role in determining creditworthiness.

---

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-card-approval.git
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook to explore and execute the code:
   ```bash
   jupyter notebook
   ```

---

## Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## Future Scope

- **Advanced Algorithms**: Incorporate techniques like XGBoost or Neural Networks for enhanced accuracy.
- **Deployment**: Deploy the model as a web application using frameworks like Flask or Django.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.



