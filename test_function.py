"""
Test heroku api response
"""
import requests
from fastapi.testclient import TestClient
from fastapi1 import app
import os
import json
import pytest

client = TestClient(app)
CSV_PATH = os.path.join(os.path.dirname(__file__), 'data_csv.csv')
JSON_PATH = os.path.join(os.path.dirname(__file__), 'data_json.json')



def test_inference():
    data_json = {
    "workclass": "Private",
    "education": "Prof-school",
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
    assert response.json() == {"PREDICTION": ">50K"}



def test_post():
    data_json = {
    "workclass": "Private",
    "education": "Prof-school",
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
    print("RESPONSE CODE: %s" % response.status_code)
    print("RESPONCE MESSAGE: %s" % response.json())


def test_get_api():
    response = client.get('https://jelal-fastapi.herokuapp.com')
    assert response.status_code == 200
    assert response.json() == 'YAHOO CENSUS PREDICTION APP IS WORKING!'




