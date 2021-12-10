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
from starter.ml.data import process_data
# Logger Configuration
from starter.ml import model as ml_model


def train_model():
    """
    Train the model on training data.
    """
    train_data = pd.read_csv('data/clean_census_tr.csv')
    train, test = train_test_split(train_data, test_size=0.2)
    
    categories = list(train.select_dtypes(include='object').columns)[:-1]
    X_train, y_train, encoder, lb = process_data(train, categories, label='salary', training=True)

    model = ml_model.train_model(X_train, y_train)


    dump(model, 'model/model.joblib')
    dump(encoder, 'model/encoder.joblib')
    dump(lb, 'model/lb.joblib')
