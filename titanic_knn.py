# knn_titanic_classification.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
df = pd.read_csv("/kaggle/input/titanic/train.csv")  # Kaggle input path

# Display first 5 rows
print("=== First 5 Rows of Titanic Dataset ===")
print(df.head().to_string(index=False))

# -----------------------------
# Step 2: Feature Selection & Preprocessing
# -----------------------------
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
target = 'Survived'

X = df[features]
y = df[target]

# Convert categorical variables to numeric
X = pd.get_dummies(X, columns=['Sex'], drop_first=True)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# -----------------------------
# Step 3: Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Step 4: Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Step 5: Train KNN Model
# -----------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# -----------------------------
# Step 6: Make Predictions
# -----------------------------
y_pred = knn.predict(X_test)

# -----------------------------
# Step 7: Evaluate Model
# -----------------------------
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Score": [
        round(accuracy_score(y_test, y_pred), 2),
        round(precision_score(y_test, y_pred), 2),
        round(recall_score(y_test, y_pred), 2),
        round(f1_score(y_test, y_pred), 2)
    ]
})

print("\n=== KNN Model Evaluation Metrics ===")
print(metrics_df.to_string(index=False))

# -----------------------------
# Step 8: Save Model as .pkl
# -----------------------------
joblib.dump(knn, "/kaggle/working/knn_titanic_model.pkl")
print("✅ KNN model saved as /kaggle/working/knn_titanic_model.pkl")
