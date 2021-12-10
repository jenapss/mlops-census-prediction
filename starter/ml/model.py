from sklearn.metrics import fbeta_score, precision_score, recall_score
import numpy as np
from joblib import load, dump
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import logging
from .data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    scores = np.array(cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro'))

    model.fit(X_train, y_train)
    logging.info('F1_macro mean %.3f' % (np.mean(scores)))
    print(np.mean(scores))
    return model

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ??? RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    model_pred = model.predict(X)
    return model_pred


def save_model(model, model_name):
    dump(model, 'model/{}.joblib'.format(model_name))

