import streamlit as st
import pandas as pd

from auto_model import select_model, train_and_predict
from llm.recommendation import (
    generate_dataset_summary,
    generate_predictions_summary,
    generate_llm_recommendation
)

# 🎨 Streamlit UI
st.set_page_config(page_title="AI-Powered Data Analysis & Recommendation System", layout="wide")
st.title("📊 AI-Powered Data Analysis & Recommendation System (with LLM)")

# 📂 File Upload
file = st.file_uploader("Upload a CSV file", type=["csv"])

# 🛠 Problem Statement Input
problem_statement = st.text_input("Describe your problem statement:")

# 🚦 Main Logic
if file and problem_statement:
    df_raw = pd.read_csv(file)
    target_column = st.selectbox("Select Target Column:", df_raw.columns)

    if st.button("Generate Recommendations"):
        st.info("🔍 Auto-selecting model, training, predicting...")

        # ✅ Auto model selection based on problem statement
        model, model_type = select_model(problem_statement)

        # ✅ Train & predict
        trained_model, predictions = train_and_predict(df_raw, target_column, model)

        st.success(f"Model Trained: {model.__class__.__name__} ({model_type})")
        st.write("✅ Sample Predictions:", pd.Series(predictions).head())

        # ✅ Generate summaries
        dataset_summary = generate_dataset_summary(df_raw, target_column)
        predictions_summary = generate_predictions_summary(model_type, predictions)

        # ✅ Call LLM for Recommendations
        st.info("🧠 Generating recommendations using LLM...")
        recommendation = generate_llm_recommendation(
            problem_statement,
            dataset_summary,
            predictions_summary
        )

        # 📊 Display
        st.subheader("📊 Dataset Summary")
        st.text(dataset_summary)

        st.subheader("📊 Predictions Summary")
        st.text(predictions_summary)

        st.subheader("💡 AI-Powered Business Recommendations:")
        st.write(recommendation)

        # 📥 Optional: Download CSV of predictions
        predictions_df = pd.DataFrame({'Predicted': predictions})
        csv = predictions_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions CSV", csv, "predictions.csv", "text/csv")
