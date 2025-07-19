from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def select_model_from_target(y):
    """
    Automatically select classification or regression model based on target column.
    """
    if y.dtype == 'object' or y.nunique() <= 10:
        logging.info("Problem detected: Classification")
        return RandomForestClassifier()
    elif np.issubdtype(y.dtype, np.number):
        logging.info("Problem detected: Regression")
        return RandomForestRegressor()
    else:
        logging.warning("Unknown target column type. Defaulting to Classification.")
        return RandomForestClassifier()


def train_and_predict(X, y, model):
    """
    Train the ML model and return predictions on the test set.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return model, pd.Series(predictions, name='Predictions')
