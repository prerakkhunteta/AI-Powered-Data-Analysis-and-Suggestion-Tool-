from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

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
   
    task_type = problem_to_task.get(problem_category.lower())

    if task_type == 'classification':
        model = RandomForestClassifier()
    elif task_type == 'regression':
        model = RandomForestRegressor()
    else:
        logging.warning(f"Unknown problem category: {problem_category}. Defaulting to classification.")
        model = RandomForestClassifier()

    return model


def train_and_predict(X, y, model):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return model, pd.Series(predictions, name='Predictions')
