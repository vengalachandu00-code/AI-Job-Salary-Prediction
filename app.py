
import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Job Salary Predictor",
    page_icon="💰",
    layout="wide"
)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("Regression_Model.pkl")
    return model


model = load_model()


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("💰 AI Job Salary Prediction")
st.write(
    "Predict the expected salary in USD based on job and company information."
)

st.divider()


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.subheader("Enter Job Details")


col1, col2 = st.columns(2)


with col1:

    job_title = st.text_input(
        "Job Title",
        value="AI Software Engineer"
    )

    experience_level = st.selectbox(
        "Experience Level",
        ["EN", "MI", "SE", "EX"]
    )

    employment_type = st.selectbox(
        "Employment Type",
        ["FT", "PT", "CT", "FL"]
    )

    company_location = st.text_input(
        "Company Location",
        value="Canada"
    )

    company_size = st.selectbox(
        "Company Size",
        ["S", "M", "L"]
    )

    employee_residence = st.text_input(
        "Employee Residence",
        value="India"
    )


with col2:

    remote_ratio = st.slider(
        "Remote Ratio (%)",
        min_value=0,
        max_value=100,
        value=50,
        step=50
    )

    required_skills = st.text_input(
        "Required Skills",
        value="Python, Machine Learning, AWS"
    )

    education_required = st.selectbox(
        "Education Required",
        ["Bachelor", "Master", "PhD", "High School"]
    )

    years_experience = st.number_input(
        "Years of Experience",
        min_value=0,
        max_value=50,
        value=2
    )

    industry = st.text_input(
        "Industry",
        value="Technology"
    )

    job_description_length = st.number_input(
        "Job Description Length",
        min_value=0,
        value=1000
    )

    benefits_score = st.number_input(
        "Benefits Score",
        min_value=0.0,
        max_value=10.0,
        value=5.0
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

st.divider()

if st.button("🔮 Predict Salary", use_container_width=True):

    # Create input dataframe
    input_data = pd.DataFrame({
    "Unnamed: 0": [0],
    "job_title": [job_title],
    "salary_currency": ["USD"],
    "experience_level": [experience_level],
    "employment_type": [employment_type],
    "company_location": [company_location],
    "company_size": [company_size],
    "employee_residence": [employee_residence],
    "remote_ratio": [remote_ratio],
    "required_skills": [required_skills],
    "education_required": [education_required],
    "years_experience": [years_experience],
    "industry": [industry],
    "posting_date": ["2024-11-20"],
    "application_deadline": ["2025-01-11"],
    "job_description_length": [job_description_length],
    "benefits_score": [benefits_score],
    "company_name": ["TechCorp Inc"]
    })

    try:

        prediction = model.predict(input_data)

        salary = prediction[0]

        st.success("Salary Prediction Completed!")

        st.metric(
            label="Predicted Salary",
            value=f"{salary:,.2f} "
        )

        st.write(
            f"### 💰 Expected Salary: {salary:,.0f} "
        )

        # Show input values
        with st.expander("View Input Details"):
            st.dataframe(input_data)

    except Exception as e:

        st.error(
            "Prediction failed. Please check that the input columns "
            "match the columns used while training the model."
        )

        st.exception(e)

