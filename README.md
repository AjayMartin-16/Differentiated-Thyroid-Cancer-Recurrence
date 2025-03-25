Differentiated Thyroid Cancer Recurrence Analysis

Project Overview

This project focuses on analyzing a dataset related to Differentiated Thyroid Cancer Recurrence to predict the likelihood of thyroid cancer recurrence based on various clinicopathologic features. The dataset consists of 13 features collected over a span of 15 years, with each patient being followed for at least 10 years.

Dataset Used

Dataset Name: Differentiated Thyroid Cancer Recurrence

Source: UC Irvine Machine Learning Repository

Size: Includes multiple records with 13 clinicopathologic features

Objective: Predict whether thyroid cancer is present or not based on the given features

Objectives

Perform Exploratory Data Analysis (EDA) to understand feature distributions and relationships.

Preprocess the dataset by handling missing values and standardizing features.

Train machine learning models to classify recurrence vs. non-recurrence cases.

Evaluate model performance using various metrics such as accuracy, precision, recall, and F1-score.

Methodology

1. Data Preprocessing

Handled missing values and checked for inconsistencies in the dataset.

Performed feature scaling and encoding for categorical variables.

Identified and removed outliers to improve model performance.

2. Exploratory Data Analysis (EDA)

Univariate Analysis: Distribution of clinical features and recurrence rates.

Bivariate Analysis: Relationship between different clinical features and recurrence.

Visualization Techniques:

Histograms and box plots to explore feature distributions.

Correlation heatmaps to identify relationships between variables.

3. Machine Learning Model Training

Implemented various classification models such as:

Logistic Regression

Decision Trees

Random Forest

Support Vector Machines (SVM)

Neural Networks (if applicable)

Used train-test splitting and cross-validation to ensure robust model evaluation.

Compared model performance using accuracy, confusion matrix, and AUC-ROC scores.

Key Findings

Certain clinical features showed a strong correlation with cancer recurrence.

Models like Random Forest and SVM performed well in predicting recurrence with high accuracy.

Feature importance analysis helped identify the most critical factors influencing recurrence.

Tools & Technologies Used

Programming Language: Python

Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, TensorFlow/Keras (if applicable)

Visualization: Matplotlib and Seaborn for graphical analysis

Business Insights & Recommendations

Early Detection Strategies: By analyzing key clinical factors, medical professionals can focus on high-risk patients for early intervention.

Optimized Treatment Plans: Predictive modeling can assist in personalizing treatment strategies to improve patient outcomes.

Further Research Opportunities: Machine learning models can be enhanced with additional patient records or deep learning techniques for better accuracy.

Future Enhancements

Implement deep learning models for improved classification performance.

Use feature engineering to extract more meaningful insights from the dataset.

Apply ensemble learning techniques to boost model performance.

Author

B. Ajay Martin Ferdinand

Acknowledgments

Dataset sourced from UC Irvine Machine Learning Repository.

Thanks to the data science and healthcare community for insights and methodologies.

GitHub Repository
