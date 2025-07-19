import streamlit as st
import pandas as pd
from data_cleaning import clean_data
from llm.problem_classifier import classify_problem
from auto_model import select_model, train_and_predict

def generate_recommendations(problem_statement, problem_category):
    statement = problem_statement.lower()

    if problem_category == 'classification':
        if 'churn' in statement:
            return "Focus on customer retention strategies like loyalty programs and personalized offers."
        elif 'pass/fail' in statement:
            return "Identify low-performing cases and suggest remedial interventions."
        elif 'fraud' in statement:
            return "Investigate high-risk cases flagged by the model."
        else:
            return "Review misclassified records and refine model thresholds."

    elif problem_category == 'regression':
        if 'price' in statement or 'sales' in statement:
            return "Adjust pricing strategy based on predicted sales trends."
        elif 'salary' in statement:
            return "Analyze salary patterns to adjust HR strategies."
        elif 'age' in statement:
            return "Target specific age groups with marketing campaigns."
        else:
            return "Focus on extreme prediction values for insights."

    return "📌 Analyze predictions to identify actionable insights for your business problem."


st.title("📊 AI Data Analysis & Recommendation System")

file = st.file_uploader("Upload a CSV file", type=["csv"])

problem_statement = st.text_input("Describe your business problem")

if "problem_category" not in st.session_state:
    st.session_state.problem_category = None
if "model" not in st.session_state:
    st.session_state.model = None
if "cleaned_features" not in st.session_state:
    st.session_state.cleaned_features = None
if "target" not in st.session_state:
    st.session_state.target = None

if file is not None:
    raw_df = pd.read_csv(file)
    st.write("Dataset Preview:")
    st.dataframe(raw_df.head())

    target_column = st.selectbox("Select Target Column:", raw_df.columns)

    if target_column:
        features_df = raw_df.drop(columns=[target_column])
        target_series = raw_df[target_column]

        cleaned_features = clean_data(features_df)

        st.session_state.cleaned_features = cleaned_features
        st.session_state.target = target_series

        

if st.button("Classify Problem"):
    if problem_statement.strip() != "":
        st.session_state.problem_category = classify_problem(problem_statement)
        st.success(f"Problem classified as: {st.session_state.problem_category}")

        st.session_state.model = select_model(st.session_state.problem_category)
        st.info(f"Selected model: {st.session_state.model.__class__.__name__}")

if st.session_state.cleaned_features is not None and st.session_state.model is not None:
    if st.button("Run Prediction"):
        with st.spinner("⏳ Training model... Please wait..."):
            trained_model, predictions = train_and_predict(
                st.session_state.cleaned_features,
                st.session_state.target,
                st.session_state.model
            )

        st.success("✅ Model trained and predictions generated.")
        

        recommendation = generate_recommendations(
            problem_statement,
            st.session_state.problem_category
        )
        st.subheader("💡 Recommended Next Steps:")
        st.write(recommendation)

        csv = predictions.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv'
        )
