import streamlit as st
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------------
# Load dataset
# -------------------------------
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
data_frame['label'] = breast_cancer_dataset.target

# Features & target
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']

# Split train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train model
model = LogisticRegression(max_iter=10000)  # increased iterations for convergence
model.fit(X_train, Y_train)

# Accuracy
train_acc = accuracy_score(Y_train, model.predict(X_train))
test_acc = accuracy_score(Y_test, model.predict(X_test))

# -------------------------------
# Streamlit App UI
# -------------------------------
st.title("🎗️ Breast Cancer Prediction App")
st.write("This app predicts whether breast cancer is **Malignant** or **Benign** using Logistic Regression.")

# Show dataset info
with st.expander("🔍 Dataset Information"):
    st.write("Number of samples:", data_frame.shape[0])
    st.write("Number of features:", data_frame.shape[1] - 1)
    st.write("Target labels: 0 = Malignant, 1 = Benign")
    st.write("Training Accuracy:", round(train_acc, 2))
    st.write("Testing Accuracy:", round(test_acc, 2))

# User Input
st.header("📊 Enter Tumor Features")

input_data = []
for feature in breast_cancer_dataset.feature_names:
    val = st.slider(
        feature,
        float(data_frame[feature].min()),
        float(data_frame[feature].max()),
        float(data_frame[feature].mean())
    )
    input_data.append(val)

# Convert to numpy
input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

# Prediction button
if st.button("🔮 Predict"):
    prediction = model.predict(input_data_as_numpy_array)
    if prediction[0] == 0:
        st.error("⚠️ The Breast Cancer is **Malignant**")
    else:
        st.success("✅ The Breast Cancer is **Benign**")
