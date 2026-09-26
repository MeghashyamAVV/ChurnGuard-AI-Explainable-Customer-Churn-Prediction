# ChurnGuard-AI-Explainable-Customer-Churn-Prediction

- cleans the data
- creates behavioral features
- one-hot encodes categorical variables
- splits train/test data
- applies SMOTE only to the training set
- trains Logistic Regression and Random Forest
- reports accuracy, precision, recall, F1, and ROC-AUC
- creates Low/Medium/High risk segments
- runs a simple retention simulation
- saves 'customer_risk_segments.csv'
- prints Random Forest feature importance
