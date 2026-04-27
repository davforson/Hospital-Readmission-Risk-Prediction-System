import streamlit as st
import requests
import json
import logging

logger = logging.getLogger(__name__)

# Page setup
st.set_page_config(
    page_title="Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Patient Readmission Risk Predictor")
st.markdown(
    "Enter patient details below to predict the probability "
    "of 30-day hospital readmission."
    )

# API configuration
API_URL = "http://localhost:8000"

# Patient information form
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("👤 Patient Demographics")

    age = st.number_input(
        "Age at Admission",
        min_value=0,
        max_value=120,
        value=65,
        help="Patient's age at the time of admission"
    )

    gender = st.selectbox(  
        "Gender role",
        options = ["Male", "Female", "Other"]
    )

    insurance = st.selectbox(
        "Insurance Type",
        options = ["Medicaire","Medicaid","Private","Self Pay","VA"]
    )

    st.subheader("🏥 Admission details")

    admission_type = st.selectbox(
        "Admission Type",
        options = ["Emergency","Urgent","Elective","Trauma"]
    )

    discharge_disposition = st.selectbox(
        "Discharge Disposition",
        options=["Home", "SNF", "Rehab", "Home Health", "AMA", "Expired"],
    )

    length_of_stay = st.number_input(
        "Length of Stay (days)",
        min_value=1,
        max_value=365,
        value=5,
    )

    primary_diagnosis = st.selectbox(
        "Primary Diagnosis",
        options=[
            "I50.9 - Heart Failure",
            "J44.1 - COPD with Exacerbation",
            "E11.9 - Type 2 Diabetes",
            "N17.9 - Acute Kidney Failure",
            "I21.9 - Acute MI",
            "J18.9 - Pneumonia",
            "K92.2 - GI Hemorrhage",
            "I63.9 - Cerebral Infarction",
            "A41.9 - Sepsis",
            "J96.0 - Acute Respiratory Failure",
        ],
        help="Select the primary ICD-10 diagnosis code",
    )

with col_right:
    st.subheader("📊 Clinical History")

    num_procedures = st.number_input(
        "Number of Procedures",
        min_value=0,
        max_value=20,
        value=2,
    )

    num_diagnoses = st.number_input(
        "Number of Diagnoses",
        min_value=1,
        max_value=20,
        value=4,
    )

    prior_admissions_6mo = st.number_input(
        "Prior Admissions (Last 6 Months)",
        min_value=0,
        max_value=20,
        value=1,
    )

    prior_admissions_12mo = st.number_input(
        "Prior Admissions (Last 12 Months)",
        min_value=0,
        max_value=30,
        value=2,
    )

    days_since_last = st.number_input(
        "Days Since Last Admission",
        min_value=0,
        max_value=3650,
        value=45,
        help="Enter 0 if this is the patient's first admission",
    )

    total_prev_los = st.number_input(
        "Total Previous Length of Stay (days)",
        min_value=0,
        max_value=365,
        value=10,
    )

    st.subheader("🔬 Lab Results")

    lab_col1, lab_col2 = st.columns(2)

    with lab_col1:
        glucose = st.number_input("Avg Glucose (mg/dL)", min_value=0.0, value=140.0, step=1.0)
        creatinine = st.number_input("Avg Creatinine (mg/dL)", min_value=0.0, value=1.2, step=0.1)
        hemoglobin = st.number_input("Avg Hemoglobin (g/dL)", min_value=0.0, value=13.5, step=0.1)
        wbc = st.number_input("Avg WBC (K/uL)", min_value=0.0, value=8.0, step=0.1)

    with lab_col2:
        bun = st.number_input("Avg BUN (mg/dL)", min_value=0.0, value=18.0, step=1.0)
        sodium = st.number_input("Avg Sodium (mEq/L)", min_value=0.0, value=140.0, step=1.0)
        potassium = st.number_input("Avg Potassium (mEq/L)", min_value=0.0, value=4.2, step=0.1)
        hba1c = st.number_input("Avg HbA1c (%)", min_value=0.0, value=6.5, step=0.1)

    num_lab_tests = st.number_input(
        "Total Lab Tests Performed",
        min_value=0,
        max_value=100,
        value=8,
    )

# Prediction button
st.markdown("---")

# Extract diagnosis code from the selection
diagnosis_code = primary_diagnosis.split(" - ")[0]

if st.button("🔍 Predict readmission risk", type='primary', use_container_width = True):

    # Build the request payload
    payload = {
        "length_of_stay": length_of_stay,
        "number_of_procedures": num_procedures,
        "number_of_diagnoses": num_diagnoses,
        "primary_diagnosis_code": diagnosis_code,
        "num_prior_admissions_6mo": prior_admissions_6mo,
        "num_prior_admissions_12mo": prior_admissions_12mo,
        "days_since_last_admission": days_since_last,
        "total_previous_los": total_prev_los,
        "avg_glucose": glucose,
        "avg_creatinine": creatinine,
        "avg_hemoglobin": hemoglobin,
        "avg_white_blood_cell": wbc,
        "avg_bun": bun,
        "avg_sodium": sodium,
        "avg_potassium": potassium,
        "avg_hba1c": hba1c,
        "num_lab_tests": num_lab_tests,
        "age_at_admission": age,
        "gender": gender,
        "insurance_type": insurance,
        "admission_type": admission_type,
        "discharge_disposition": discharge_disposition
    }

    try:
        # Call the FastAPI prediction Endpoint
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()
            probability = result["readmission_probability"]
            risk_level = result["risk_level"]
            model_version = result["model_version"]

            # Display results
            st.markdown("---")
            st.subheader("Prediction Result")

            # Risk level with color coding
            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:
                if risk_level == "high":
                    st.error(f"🔴 High risk")
                elif risk_level == "medium":
                    st.warning(f"🟡 Medium risk")
                else: 
                    st.success(f"🟢 LOW RISK")

            
            with result_col2:
                st.metric(
                    "Readmission Probability",
                    f"{probability * 100: .1f}%"
                )

            with result_col3:
                st.metric(
                    "Model version",
                    model_version
                )

            # Recommendation based on risk level
            st.markdown("### 📋 Recommended Actions")
            if risk_level == "high":
                st.markdown(
                    "- Schedule follow-up appointment within **7 days** of discharge\n"
                    "- Arrange **home health services**\n"
                    "- Conduct thorough **medication reconciliation**\n"
                    "- Contact patient's primary care physician\n"
                    "- Consider **social work consultation** for support services"
                )
            elif risk_level == "medium":
                st.markdown(
                    "- Schedule follow-up appointment within **14 days** of discharge\n"
                    "- Provide detailed **discharge instructions**\n"
                    "- Ensure **medication understanding** before discharge\n"
                    "- Confirm patient has **transportation** to follow-up"
                )
            else:
                st.markdown(
                    "- Schedule standard follow-up appointment\n"
                    "- Provide **standard discharge instructions**\n"
                    "- Encourage patient to call with any concerns"
                )

        else:
            st.error(f"API Error: {response.status_code} - {response.text} ")


    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the prediction API. "
            "Make sure the API is running: `uvicorn api.main:app --port 8000`"
        )

    except requests.exceptions.Timeout:
        st.error("Request timed out. The API may be overloaded.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# ── Footer ──
st.markdown("---")
st.caption(
    "This tool provides a risk estimate to support clinical decision-making. "
    "It is not a substitute for clinical judgment."
)



