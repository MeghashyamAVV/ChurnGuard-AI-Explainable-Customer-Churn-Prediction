import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from imblearn.over_sampling import SMOTE

DATA_PATH = "data/customer_churn.csv"

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna().copy()

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

df["AverageMonthlySpend"] = (
    df["TotalCharges"] /
    df["tenure"].replace(0, 1)
)

df["IsLongTermCustomer"] = (
    df["tenure"] >= 24
).astype(int)

df["IsHighMonthlySpend"] = (
    df["MonthlyCharges"] >
    df["MonthlyCharges"].median()
).astype(int)

df["HasSecurity"] = (
    df["OnlineSecurity"] == "Yes"
).astype(int)

df["HasTechSupport"] = (
    df["TechSupport"] == "Yes"
).astype(int)

df["HasOnlineBackup"] = (
    df["OnlineBackup"] == "Yes"
).astype(int)

df["HasDeviceProtection"] = (
    df["DeviceProtection"] == "Yes"
).astype(int)

df["HasDependents"] = (
    df["Dependents"] == "Yes"
).astype(int)

df["HasPartner"] = (
    df["Partner"] == "Yes"
).astype(int)

df = df.drop(columns=["customerID"])

X = df.drop(columns=["Churn"])
y = df["Churn"]

# Convert categorical columns to numerical columns
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nClass distribution before SMOTE:")
print(y_train.value_counts())

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("\nClass distribution after SMOTE:")
print(pd.Series(y_train_smote).value_counts())

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(
    X_train_smote,
    y_train_smote
)

y_pred_logistic = logistic_model.predict(X_test_scaled)
y_prob_logistic = logistic_model.predict_proba(
    X_test_scaled
)[:, 1]

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(
    X_train_smote,
    y_train_smote
)

y_pred_rf = rf_model.predict(X_test_scaled)
y_prob_rf = rf_model.predict_proba(
    X_test_scaled
)[:, 1]

def evaluate_model(name, y_true, y_pred, y_prob):
    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print("Accuracy :", round(accuracy_score(y_true, y_pred), 4))
    print("Precision:", round(precision_score(y_true, y_pred), 4))
    print("Recall   :", round(recall_score(y_true, y_pred), 4))
    print("F1       :", round(f1_score(y_true, y_pred), 4))
    print("ROC-AUC  :", round(roc_auc_score(y_true, y_prob), 4))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


evaluate_model(
    "Logistic Regression",
    y_test,
    y_pred_logistic,
    y_prob_logistic
)

evaluate_model(
    "Random Forest",
    y_test,
    y_pred_rf,
    y_prob_rf
)

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy_score(y_test, y_pred_logistic),
        accuracy_score(y_test, y_pred_rf)
    ],
    "Precision": [
        precision_score(y_test, y_pred_logistic),
        precision_score(y_test, y_pred_rf)
    ],
    "Recall": [
        recall_score(y_test, y_pred_logistic),
        recall_score(y_test, y_pred_rf)
    ],
    "F1": [
        f1_score(y_test, y_pred_logistic),
        f1_score(y_test, y_pred_rf)
    ],
    "ROC-AUC": [
        roc_auc_score(y_test, y_prob_logistic),
        roc_auc_score(y_test, y_prob_rf)
    ]
})

print("\nModel comparison:")
print(results.round(4))

risk_df = pd.DataFrame({
    "ChurnProbability": y_prob_rf,
    "ActualChurn": y_test.values
})

def risk_category(probability):
    if probability < 0.30:
        return "Low Risk"
    elif probability < 0.60:
        return "Medium Risk"
    else:
        return "High Risk"

risk_df["RiskSegment"] = risk_df[
    "ChurnProbability"
].apply(risk_category)

print("\nRisk segment counts:")
print(risk_df["RiskSegment"].value_counts())

print("\nAverage churn probability by segment:")
print(
    risk_df.groupby("RiskSegment")["ChurnProbability"]
    .mean()
    .round(4)
)

high_risk = risk_df[
    risk_df["RiskSegment"] == "High Risk"
]

expected_churn = high_risk["ChurnProbability"].sum()
retention_effectiveness = 0.20

prevented_churn = (
    expected_churn *
    retention_effectiveness
)

print("\nRetention simulation:")
print("High-risk customers:", len(high_risk))
print("Expected churn:", round(expected_churn, 2))
print(
    "Estimated prevented churn:",
    round(prevented_churn, 2)
)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\nTop 15 feature importances:")
print(importance.head(15).round(4))

risk_df.to_csv(
    "customer_risk_segments.csv",
    index=False
)

importance.to_csv(
    "feature_importance.csv",
    index=False
)

print("\nSaved:")
print("- customer_risk_segments.csv")
print("- feature_importance.csv")