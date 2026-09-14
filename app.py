import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "churn_model.pkl"

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📊 Customer Churn Prediction")

st.write(
    "Predict whether a telecom customer is likely to churn "
    "using a machine learning model."
)

st.divider()


# --------------------------------------------------
# CUSTOMER INFORMATION
# --------------------------------------------------

st.header("👤 Customer Information")

col1, col2, col3 = st.columns(3)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )


with col2:

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )


with col3:

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )


# --------------------------------------------------
# BILLING INFORMATION
# --------------------------------------------------

st.header("💳 Billing Information")

col1, col2, col3 = st.columns(3)


with col1:

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )


with col2:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


with col3:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )


total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=float(monthly_charges * tenure)
)


st.divider()


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button(
    "🔮 Predict Churn",
    use_container_width=True
):

    input_data = pd.DataFrame({

        "gender": [gender],

        "SeniorCitizen": [senior_citizen],

        "Partner": [partner],

        "Dependents": [dependents],

        "tenure": [tenure],

        "PhoneService": [phone_service],

        "MultipleLines": [multiple_lines],

        "InternetService": [internet_service],

        "OnlineSecurity": [online_security],

        "OnlineBackup": [online_backup],

        "DeviceProtection": [device_protection],

        "TechSupport": [tech_support],

        "StreamingTV": [streaming_tv],

        "StreamingMovies": [streaming_movies],

        "Contract": [contract],

        "PaperlessBilling": [paperless_billing],

        "PaymentMethod": [payment_method],

        "MonthlyCharges": [monthly_charges],

        "TotalCharges": [total_charges]
    })


    # ----------------------------------------------
    # MODEL PREDICTION
    # ----------------------------------------------

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    classes = list(model.classes_)


    # Automatically identify churn class
    if "Yes" in classes:
        churn_class = "Yes"

    elif 1 in classes:
        churn_class = 1

    else:
        st.error(
            f"Could not identify churn class. Model classes: {classes}"
        )
        st.stop()


    churn_index = classes.index(churn_class)

    churn_probability = probabilities[churn_index]

    probability_percent = churn_probability * 100


    # ----------------------------------------------
    # RESULT
    # ----------------------------------------------

    st.subheader("🔮 Prediction Result")


    if prediction == churn_class:

        st.error("⚠️ High Churn Risk")

        st.metric(
            "Churn Probability",
            f"{probability_percent:.2f}%"
        )

        st.warning(
            "This customer is predicted to be at risk of leaving."
        )


        st.write("### 💡 Recommended Actions")

        st.write(
            "• Offer a personalized retention discount"
        )

        st.write(
            "• Consider upgrading the customer's contract"
        )

        st.write(
            "• Provide technical support assistance"
        )

        st.write(
            "• Offer loyalty benefits"
        )


    else:

        st.success("✅ Low Churn Risk")

        st.metric(
            "Churn Probability",
            f"{probability_percent:.2f}%"
        )

        st.info(
            "This customer is predicted to remain with the company."
        )


    # ----------------------------------------------
    # CUSTOMER DATA
    # ----------------------------------------------

    st.write("### 📋 Customer Data")

    st.dataframe(
        input_data,
        use_container_width=True
    )
