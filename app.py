import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load and prepare the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Train a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Streamlit app
st.title("Iris Flower Classification")

# User input for feature values
sepal_length = st.text_input('Sepal Length (cm)', value=f"{X[:, 0].mean()}")
sepal_width = st.text_input('Sepal Width (cm)', value=f"{X[:, 1].mean()}")
petal_length = st.text_input('Petal Length (cm)', value=f"{X[:, 2].mean()}")
petal_width = st.text_input('Petal Width (cm)', value=f"{X[:, 3].mean()}")

# Convert text input to float
try:
    sepal_length = float(sepal_length)
    sepal_width = float(sepal_width)
    petal_length = float(petal_length)
    petal_width = float(petal_width)
except ValueError:
    st.error("Please enter valid numerical values.")
    st.stop()

# Prepare the input data for prediction
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=feature_names)
input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = clf.predict(input_data_scaled)
prediction_proba = clf.predict_proba(input_data_scaled)

# Show prediction result
st.write("Prediction:")
st.write(f"Species: {target_names[prediction][0]}")

st.write("Prediction Probability:")
for i, species in enumerate(target_names):
    st.write(f"{species}: {prediction_proba[0][i]:.2f}")
