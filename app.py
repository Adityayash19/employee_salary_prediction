import streamlit as st
import pandas as pd
import joblib
import time

# Load trained model
model = joblib.load('best_model.pkl')

# Page config
st.set_page_config(page_title="Employee Salary Classification", layout="wide")

# Custom CSS for dark theme styling
st.markdown("""
    <style>
        body {
            background-color: #0e1117;
        }
        .css-18e3th9 {
            background-color: #0e1117;
        }
        .stButton>button {
            color: white;
            background-color: #6c63ff;
            border-radius: 5px;
        }
        .st-cd {
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar - Input
st.sidebar.header("🎯 Input Employee Details")

age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education Level", ['Bachelors', 'HS-grad', 'Masters', 'Some-college', 'Assoc-acdm', 'Assoc-voc', 'Doctorate'])
occupation = st.sidebar.selectbox("Job Role", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Machine-op-inspct', 'Adm-clerical'])

hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

# Additional required inputs
workclass = st.sidebar.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])

fnlwgt = st.sidebar.number_input("**Final Weight (FNLWGT) – (OPTIONAL)**", value=100000)
education_num = st.sidebar.slider("**Educational Num – (OPTIONAL)**", 1, 16, 10)

marital_status = st.sidebar.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent'])
relationship = st.sidebar.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
race = st.sidebar.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
native_country = st.sidebar.selectbox("Native Country", ['United-States', 'India', 'Mexico', 'Philippines', 'Germany', 'Canada', 'England', 'Other'])

capital_gain = st.sidebar.number_input("**Capital Gain – (OPTIONAL)**", value=0)
capital_loss = st.sidebar.number_input("**Capital Loss – (OPTIONAL)**", value=0)

# Main Heading
st.markdown("## 💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Prepare input
input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours-per-week': [hours_per_week],
    'experience': [experience],
    'workclass': [workclass],
    'fnlwgt': [fnlwgt],
    'educational-num': [education_num],
    'marital-status': [marital_status],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'native-country': [native_country],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
})

# Show input
st.subheader("🔍 Input Data")
st.write(input_df)

# Prediction with progress bar
if st.button("Predict Salary Class"):
    with st.spinner("Analyzing Data..."):
        progress = st.progress(0)
        for percent in range(1, 101):
            time.sleep(0.01)
            progress.progress(percent)

        try:
            prediction = model.predict(input_df)[0]
            st.success(f"✅ Prediction: Employee earns {'>50K' if prediction == 1 else '≤50K'}")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

# Footer with Linktree and credit
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <a href="https://linktr.ee/AdityaYash19" target="_blank" style="font-size:18px;">🔗 Check My Linktree</a><br>
        <p style="margin-top:10px; color: #888;">MADE BY <strong>ADITYA YASH</strong> WITH ❤️</p>
    </div>
    """, unsafe_allow_html=True
)
