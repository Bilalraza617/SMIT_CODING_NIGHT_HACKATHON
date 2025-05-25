import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and feature list
model = joblib.load('spacex_model.pkl')
model_features = joblib.load('model_features.pkl')

st.title("🚀 SpaceX Launch Success Prediction App")

st.sidebar.header("🧮 Input Parameters")

# User Inputs
flight_number = st.sidebar.number_input("Flight Number", min_value=1, max_value=300, step=1)
payload_mass_kg = st.sidebar.number_input("Payload Mass (kg)", min_value=0, max_value=10000, step=100)

rocket_choice = st.sidebar.selectbox("Rocket", ['Falcon 9', 'Falcon Heavy'])
launch_site_choice = st.sidebar.selectbox("Launch Site", ['CCAFS SLC 40', 'KSC LC 39A', 'VAFB SLC 4E'])
landing_type = st.sidebar.selectbox("Landing Type", ['ASDS', 'RTLS', 'Ocean', 'unknown'])
core_reused = st.sidebar.checkbox("Core Reused")

# Initialize input dictionary
input_dict = {
    'flight_number': flight_number,
    'payload_mass_kg': payload_mass_kg,
    f'rocket_{rocket_choice}': 1,
    f'launch_site_{launch_site_choice}': 1,
    f'landing_type_{landing_type}': 1,
    'core_reused_True': int(core_reused)
}

# Create DataFrame and align with model features
input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=model_features, fill_value=0)

# Prediction
if st.button("Predict Launch Outcome"):
    pred = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.success(f"✅ Successful Launch Predicted with {round(pred_proba * 100, 2)}% confidence.")
    else:
        st.error(f"❌ Launch Failure Predicted with {round((1 - pred_proba) * 100, 2)}% confidence.")

# Optional Dashboard Section
st.markdown("---")
st.subheader("📊 Sample Launch Data Overview (Optional)")

# Load sample dataset (if available)
try:
    df = pd.read_csv('cleaned_data_spacex_h.csv')  # You must have this ready!
    
    # Pie chart
    success_counts = df['success'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(success_counts, labels=success_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

    # Correlation heatmap
    st.write("📈 Correlation Heatmap")
    corr = df.select_dtypes(include='number').corr()
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.heatmap(corr, annot=True, cmap='Blues', ax=ax2)
    st.pyplot(fig2)

except FileNotFoundError:
    st.warning("🔍 'cleaned_launch_data.csv' not found. Dashboard visuals won't be shown.")
