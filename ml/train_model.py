import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


# --------------------------------
# 1. Load dataset
# --------------------------------

df = pd.read_csv(
    "../data/historical_task_assignment_5000.csv"
)

print("Dataset loaded")
print(df.shape)


# --------------------------------
# 2. Select features
# --------------------------------

features = [
    "Skill_Match_Percent",
    "Experience_Years",
    "Employee_Workload_Percent",
    "Employee_Availability_Percent",
    "Past_Performance_Score",
    "Story_Points",
    "Expected_Hours"
]

X = df[features]

y = df["Assignment_Success"]


# --------------------------------
# 3. Split dataset
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# --------------------------------
# 4. Create XGBoost model
# --------------------------------

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)


# --------------------------------
# 5. Train
# --------------------------------

model.fit(
    X_train,
    y_train
)


print("Model training completed")


# --------------------------------
# 6. Predict test data
# --------------------------------

predictions = model.predict(X_test)


# --------------------------------
# 7. Evaluate
# --------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)


# --------------------------------
# 8. Save model
# --------------------------------

joblib.dump(
    model,
    "model/developer_model.pkl"
)

print("\nModel saved!")