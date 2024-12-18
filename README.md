# Gestational-diabetes-prediction
This repository hosts the complete implementation of a deep learning pipeline for Gestational Diabetes Mellitus prediction, utilizing a hybrid CNN-LSTM architecture with attention mechanisms. The project includes data preprocessing, feature engineering, model implementation, and comprehensive evaluation metrics, achieving state-of-the-art results.

Project Overview
Gestational Diabetes Mellitus is a critical condition affecting pregnant women. This project leverages Convolutional Neural Networks (CNNs) for spatial feature extraction and Long Short-Term Memory (LSTM) networks for capturing temporal dependencies in clinical datasets. The model is trained and evaluated on preprocessed data, achieving 97% accuracy, surpassing traditional machine learning models like Logistic Regression, Random Forests, and XGBoost.

Dataset Details
Source: Clinical datasets with 47 features, including glucose levels, BMI, and patient demographics.

Preprocessing Steps:
Handling missing values using median and mode imputation.
Removing duplicate and constant columns.
Balancing classes using SMOTE to address class imbalance.

Model Architecture
CNN-LSTM Hybrid Model
Convolutional Layers: Extract spatial relationships between features like glucose levels and BMI.
LSTM Layers: Analyze sequential data, capturing trends over trimesters.
Attention Mechanism: Highlights the most critical features for prediction.

Evaluation Metrics
Accuracy: 97%
Precision: 95%
Recall: 92%
ROC-AUC: 98%

Key Features
Advanced data preprocessing pipeline to handle real-world clinical data issues.
Class balancing using SMOTE to address dataset imbalance.
A robust CNN-LSTM architecture with attention mechanisms.
Comprehensive evaluation and visualization of results.
Deployment-ready architecture for real-time clinical use.

Requirements
Python 3.8+
Libraries: TensorFlow, Keras, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn

Results
The CNN-LSTM model demonstrated superior performance:

Accuracy: 97%
ROC-AUC: 98%
Comparison with other models:
Logistic Regression: 75%
Random Forest: 82%
XGBoost: 87%

Future Work
Integration with electronic health records for real-time predictions.
Deployment of the model using Docker and Flask for clinical use.
Exploring additional temporal features to enhance predictive performance.
