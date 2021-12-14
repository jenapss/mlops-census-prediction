
import requests
from fastapi.testclient import TestClient
from fastapi1 import app
import os
import json
import pytest

client = TestClient(app)


def test_live_api():
    data_json = {
    "workclass": "State-gov",
    "education": "Bachelors",
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Unmarried",
    "race": "White",
    "sex": "Male",
    "native-country": "United-States",
    "age": 35,
    "hours-per-week": 40,

}

    response = requests.post(
        'https://jelal-fastapi.herokuapp.com/predict', json=data_json)

    assert response.status_code == 200
    return response.json() 