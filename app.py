import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="UK Rental Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# -------------------- Custom Styling --------------------
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
}
.main {
    background-color: #f6f8fb;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
.hero {
    background: linear-gradient(135deg, #16324f, #2f6f9f);
    padding: 2rem 2rem 1.6rem 2rem;
    border-radius: 20px;
    color: white;
    margin-bottom: 1.2rem;
}
.hero h1 {
    margin-bottom: 0.4rem;
}
.card {
    background: white;
    padding: 1.4rem;
    border-radius: 18px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    margin-bottom: 1rem;
}
.result-card {
    background: linear-gradient(135deg, #ecfdf3, #dff5e8);
    border-left: 6px solid #2e7d32;
    padding: 1.2rem;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}
.result-amount {
    font-size: 2rem;
    font-weight: 700;
    color: #145a32;
    margin-top: 0.4rem;
}
.label-muted {
    color: #5f6b7a;
    font-size: 0.95rem;
}
.small-note {
    color: #6b7280;
    font-size: 0.9rem;
}
.sidebar-title {
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 0.8rem;
}
hr {
    margin-top: 1rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------- Load Artifacts --------------------
model = joblib.load("rental_price_model.pkl")
scaler = joblib.load("scaler.pkl")
model_features = joblib.load("rental_model_features.pkl")


def predict_rent(city: str, rooms: int, property_type: str) -> float:
    city = city.strip().title()
    property_type = property_type.strip().title()

    # Optional mapping
    if property_type == "Apartment":
        property_type = "Flat"

    input_encoded = pd.DataFrame(0, index=[0], columns=model_features)
    input_encoded.loc[0, "Number of Rooms"] = rooms

    city_col = f"City_{city}"
    if city_col in input_encoded.columns:
        input_encoded.loc[0, city_col] = 1

    property_col = f"Property_Type_{property_type}"
    if property_col in input_encoded.columns:
        input_encoded.loc[0, property_col] = 1

    input_scaled = scaler.transform(input_encoded)
    log_prediction = model.predict(input_scaled)
    predicted_price = np.exp(log_prediction)

    return float(predicted_price[0])


available_cities = sorted(
    [col.replace("City_", "") for col in model_features if col.startswith("City_")]
)

available_property_types = sorted(
    [col.replace("Property_Type_", "") for col in model_features if col.startswith("Property_Type_")]
)

# -------------------- Header --------------------
st.markdown("""
<div class="hero">
    <h1>🏠 UK Rental Price Predictor</h1>
    <p>Estimate monthly rental prices using a trained machine learning model based on city, number of rooms, and property type.</p>
</div>
""", unsafe_allow_html=True)

# -------------------- Sidebar Form --------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">Enter Property Details</div>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        city_input = st.selectbox("City", available_cities)
        rooms_input = st.number_input(
            "Number of Rooms",
            min_value=1,
            max_value=10,
            value=2,
            step=1
        )
        property_type_input = st.selectbox("Property Type", available_property_types)
        submitted = st.form_submit_button("Predict Rent", use_container_width=True)

# -------------------- Main Layout --------------------
col1, col2 = st.columns([1.15, 0.85])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Property Summary")
    st.write("Use the form in the sidebar to choose the property details for prediction.")
    summary_df = pd.DataFrame({
        "Feature": ["City", "Number of Rooms", "Property Type"],
        "Value": [city_input, rooms_input, property_type_input]
    })
    st.table(summary_df)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("How the Prediction Works")
    st.write(
        "The system uses the saved best-performing machine learning model from the project. "
        "Your inputs are encoded, scaled using the saved preprocessing pipeline, and then passed "
        "to the trained model to estimate the monthly rental price."
    )
    st.markdown(
        '<p class="small-note">Predictions are estimates based on the available training dataset and selected features.</p>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction Output")

    if submitted:
        prediction = predict_rent(city_input, rooms_input, property_type_input)

        st.markdown(f"""
        <div class="result-card">
            <div class="label-muted">Estimated Monthly Rent</div>
            <div class="result-amount">£{prediction:,.2f}</div>
            <hr>
            <div class="label-muted">
                {rooms_input}-bedroom {property_type_input} in {city_input}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.metric("Predicted Rent", f"£{prediction:,.2f}")
    else:
        st.info("Fill in the property details in the sidebar and click **Predict Rent**.")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Footer --------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Project Context")
st.write(
    "This app was developed as part of a Computing Masters Project on the design and evaluation "
    "of a machine learning-based system for predicting rental property prices in the UK housing market."
)
st.markdown('</div>', unsafe_allow_html=True)