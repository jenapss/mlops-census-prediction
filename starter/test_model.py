import pandas as pd
from joblib import load
import numpy as np
import pytest
import os

from starter.ml import model as ml_model
from starter.ml.data import process_data

MODEL_DIR =os.path.join(os.path.dirname(__file__),
    '..',
    'model','model.joblib')
ENCODER_DIR = os.path.join(os.path.dirname(__file__),
    '..',
    'model','encoder.joblib')
LB_DIR = os.path.join(os.path.dirname(__file__),
    '..',
    'model','lb.joblib')


@pytest.fixture
def random_forest():
    return load(MODEL_DIR)


@pytest.fixture
def binarizer():
    return load(LB_DIR)


@pytest.fixture
def encoder():
    return load(ENCODER_DIR)


@pytest.fixture
def data():
    """
    Get the training data
    """
    try:
        df = pd.read_csv(os.path.join(
            os.getcwd(), "starter/data/clean_census_tr.csv"))
    except FileNotFoundError:
        df = pd.read_csv(os.path.join(
            os.getcwd(), "data/clean_census_ts.csv"))

    return df


def test_process_data(encoder, binarizer, data):
    cats = list(data.select_dtypes(include='object').columns)[:-1]

    X, y, _, _ = process_data(data,
                              categorical_features=cats,
                              label='salary', training=False, encoder=encoder,
                              lb=binarizer)
    assert isinstance(X, np.ndarray)
    assert len(X) > 0
    assert isinstance(y, np.ndarray)


def test_model1():

    temp_dataset = pd.DataFrame({
        "var1":[1,2,-3,-1,2,2,3,4,1,3],
        "var2":[0,0,0,1,1,3,4,1,2,5],
        "var3":[2.7,1.5,-0.8,0.2,-2.1,-3.3,-4.4,-5.3, 1.1,1.2],
        "label":[1,1,1,1,1,0,0,0,0,0]
    })

    X = temp_dataset
    y = X.pop("label")

    model = ml_model.train_model(X, y)

    assert model.n_classes_ == 2

def test_model_performance():
    temp_dataset = pd.DataFrame({
        "var1":[1,2,-3,-1,2,2,3,4,1,3],
        "var2":[0,0,0,1,1,3,4,1,2,5],
        "var3":[2.7,1.5,-0.8,0.2,-2.1,-3.3,-4.4,-5.3, 1.1,1.2],
        "label":[1,1,1,1,1,0,0,0,0,0]
    })
    X = temp_dataset
    y = X.pop("label")
    model = ml_model.train_model(X, y)

    assert all(ml_model.inference(model, X) == [1,1,1,1,1,0,0,0,0,0])