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
    
    ml_model.save_model(model, 'model')
    ml_model.save_model(encoder, 'encoder')
    ml_model.save_model(lb, 'lb')