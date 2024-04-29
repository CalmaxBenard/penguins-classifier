import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="classifier model", page_icon="📊", layout="wide")
st.header("Penguins Classification Predictor")
st.write("""
This app predicts the **Palmer Penguin** species.
Data obtained from the [palmerpenguins library](https://github.com//allisonhorst/palmerpenguins) in R by Allison Horst.
""")

st.sidebar.header("User Input Features")

st.sidebar.markdown("[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master"
                    "/penguins_example.csv)")

# Collect user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["CSV", "txt"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox("Island", ("Biscoe", "Dream", "Torgersen"))
        sex = st.sidebar.selectbox("Sex", ("MALE", "FEMALE"))
        bill_length_mm = st.sidebar.slider("Bill length (mm)", 32.1, 59.6, 43.9)
        bill_depth_mm = st.sidebar.slider("Bill depth (mm)", 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider("Flipper length (mm)", 172.0, 231.0, 201.0)
        body_mass_g = st.sidebar.slider("Body mass (g)", 2700.0, 6300.0, 4207.0)
        data = {
            "island": island,
            "culmen_length_mm": bill_length_mm,
            "culmen_depth_mm": bill_depth_mm,
            "flipper_length_mm": flipper_length_mm,
            "body_mass_g": body_mass_g,
            "sex": sex
        }
        features = pd.DataFrame(data, index=[0])
        return features


    input_df = user_input_features()

# Combine user input features with entire penguins dataset
penguins_raw = pd.read_csv("penguins_clean.csv")
penguins = penguins_raw.drop("species", axis=1)
df = pd.concat([input_df, penguins], axis=0)

# Encoding of Ordinal features
encode = ["sex", "island"]
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col) * 1
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1]  # selects only the first row (user input data)

# Display user input features
st.subheader("User Input Features")

if uploaded_file is not None:
    st.write(df)
else:
    st.write("Awaiting CSV file to be uploaded. Currently using input parameters")
    st.write(df)

# Read in saved classification model
loaded_model = pickle.load(open("penguins_clf.pkl", "rb"))

# Apply model to make predictions
prediction = loaded_model.predict(df)
prediction_proba = loaded_model.predict_proba(df)

col1, col2 = st.columns(2)
# Display predictions
with col1:
    st.subheader("Penguin Species Prediction")
    penguins_species = np.array(["Adelie", "Chinstrap", "Gentoo"])
    st.write(penguins_species[prediction])

with col2:
    st.subheader("Prediction Probability")
    st.write(prediction_proba)
