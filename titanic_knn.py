import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

st.title("Titanic KNN Survival Predictor")

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
file_path = "train.csv"  # Dataset in repo root
if not os.path.exists(file_path):
    st.error(f"{file_path} not found. Please add the Titanic dataset to the repo root.")
    st.stop()

df = pd.read_csv(file_path)
st.subheader("First 5 Rows of Dataset")
st.dataframe(df.head())

# -----------------------------
# Step 2: Preprocessing & Model Training
# -----------------------------
# Only use features user can reasonably know
features = ['Pclass', 'Sex', 'Age']
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

# Save model
joblib.dump(knn, "knn_titanic_model.pkl")
st.success("✅ KNN model trained and saved as knn_titanic_model.pkl")

# -----------------------------
# Step 3: User Input for Prediction
# -----------------------------
st.subheader("Predict Survival for a Passenger")

# Only inputs the user realistically knows
pclass = st.selectbox("Passenger Class (1 = First, 2 = Second, 3 = Third)", [1, 2, 3])
sex = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age of Passenger", 0, 100, 30)

# Prepare input for model
input_df = pd.DataFrame({
    'Pclass': [pclass],
    'Age': [age],
    'Sex_male': [1 if sex.lower() == 'male' else 0]
})

# Match columns
for col in X.columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Scale input
input_scaled = scaler.transform(input_df[X.columns])

# Predict
prediction = knn.predict(input_scaled)[0]
probability = knn.predict_proba(input_scaled)[0][prediction]

st.subheader("Prediction Result")
st.write(f"Predicted Survival: **{'Survived' if prediction==1 else 'Did Not Survive'}**")
st.write(f"Prediction Probability: **{probability:.2f}**")
