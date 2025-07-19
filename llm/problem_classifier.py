import difflib
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load API token from .env
load_dotenv()

# Initialize Hugging Face Endpoint
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational"
)
model = ChatHuggingFace(llm=llm)

# Define Categories
labels = [
    "sales decline",
    "customer churn",
    "fraud detection",
    "employee retention",
    "supply chain issue",
    "financial loss",
    "low productivity",
    "customer complaints",
    "market competition",
    "data security concern",
    "regulatory compliance risk"
]

# Prompt Template
template = """
You are an AI classifier.

Given the following problem statement:
"{problem_statement}"

Categories:
{labels}

Rules:
- Choose ONLY ONE category from the above.
- Do NOT explain your answer.
- Respond strictly in this format:

Category: <category_name>
"""

prompt = PromptTemplate.from_template(template)

# 🔍 Problem Classifier Function
def classify_problem(problem_statement):
    prompt_text = prompt.format(
        problem_statement=problem_statement,
        labels=", ".join(labels)
    )
    response = model.invoke(prompt_text).content.strip()

    # Extract category
    if "Category:" in response:
        predicted = response.split("Category:")[-1].strip()
    else:
        predicted = response.strip()

    # Fuzzy match for safety
    closest_match = difflib.get_close_matches(predicted.lower(), [l.lower() for l in labels], n=1, cutoff=0.5)
    if closest_match:
        matched_index = [l.lower() for l in labels].index(closest_match[0])
        return labels[matched_index]
    else:
        return "Other"

# For standalone testing
if __name__ == "__main__":
    example_problem = "high customer loss in last quarter"
    result = classify_problem(example_problem)
    print(f"Classified Problem: {result}")
