# auto_model.py

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from data_cleaning import clean_data

problem_to_task = {
    'sales decline': 'regression',
    'customer churn': 'classification',
    'fraud detection': 'classification',
    'employee retention': 'classification',
    'supply chain issue': 'regression',
    'financial loss': 'regression',
    'low productivity': 'regression',
    'customer complaints': 'classification',
    'market competition': 'classification',
    'data security concern': 'classification',
    'regulatory compliance risk': 'classification'
}

def select_model(problem_category):
    task_type = problem_to_task.get(problem_category.lower(), 'classification')
    if task_type == 'classification':
        return RandomForestClassifier(), 'classification'
    else:
        return RandomForestRegressor(), 'regression'


def train_and_predict(df, target_column, model):
    # Separate features and target
    features_df = df.drop(columns=[target_column])
    target = df[target_column]

    # Clean features using your clean_data function
    cleaned_features_df = clean_data(features_df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(cleaned_features_df, target, test_size=0.2, random_state=42)

    # Train model
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    return model, predictions

