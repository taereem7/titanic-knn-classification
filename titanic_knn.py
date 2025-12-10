import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

st.title("Titanic KNN Classification")

# Load dataset
file_path = "train.csv"
if not os.path.exists(file_path):
    st.error(f"{file_path} not found. Please upload the dataset in the repo root.")
    st.stop()

df = pd.read_csv(file_path)

# Display first 5 rows
st.subheader("First 5 rows of the dataset")
st.dataframe(df.head())

# Preprocessing
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
target = 'Survived'

X = pd.get_dummies(df[features], drop_first=True)
y = df[target]
X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)

# Split & scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predictions & metrics
y_pred = knn.predict(X_test)
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Score": [
        round(accuracy_score(y_test, y_pred), 2),
        round(precision_score(y_test, y_pred), 2),
        round(recall_score(y_test, y_pred), 2),
        round(f1_score(y_test, y_pred), 2)
    ]
})

st.subheader("KNN Model Evaluation Metrics")
st.table(metrics_df)

# Save model
joblib.dump(knn, "knn_titanic_model.pkl")
st.success("✅ KNN model saved as knn_titanic_model.pkl")
