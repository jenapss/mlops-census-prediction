# Script to train machine learning model.


# Add the necessary imports for the starter code.
import pandas as pd
import os
import logging
import yaml
import json
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import starter
from starter.ml.data import process_data, load_data
# Logger Configuration
from starter.ml import model as ml_model


def train_model():

    train, test = load_data('data/clean_census_tr.csv')
    categories = list(train.select_dtypes(include='object').columns)[:-1]
    X, y, encoder, lb = process_data(train, categories, label='salary', training=True)

    #train model
    model = ml_model.train_model(X, y)


    # Proces the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test, categorical_features=categories, label='salary', training=False, encoder=encoder, lb=lb)
    y_test_preds = ml_model.inference(model,X_test )
    precision, recall, fbeta = ml_model.compute_model_metrics(y_test, y_test_preds)

    print(' METRICS ON TEST DATA --- PRECISION = {}  RECAL = {}  FBETA = {}'.format(precision, recall, fbeta))
    ml_model.save_model(model, 'model')
    ml_model.save_model(encoder, 'encoder')
    ml_model.save_model(lb, 'lb')