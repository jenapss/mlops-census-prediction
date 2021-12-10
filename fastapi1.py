from fastapi import FastAPI
from joblib import load
from pandas.core.frame import DataFrame
import starter.ml.model
from pydantic import BaseModel, Field
from starter import train_model, slice_score
from starter.ml import model as model_package
from starter.ml import data as data_package

import os

class Census(BaseModel):
    workclass: str = Field(..., example = "Never-married")
    education: str = Field(..., example= 'Bachelors')
    marital_status: str = Field(..., alias = 'marital-status')
    occupation: str = Field(..., example = 'Adm-clerical')
    relationship: str = Field(..., example = 'husband')
    race: str = Field(..., example = 'White')
    sex: str = Field(..., example = 'Male')
    native_country: str = Field(..., alias = 'native-country', 
                                    example = 'Adm-clerical')
    age: int = Field(..., example = 35)
    hours_per_week: int = Field(..., alias = 'hours-per-week',
                                     example = 45)

app = FastAPI()

@app.get('/')
async def get():
    return {'message': 'Hello'}


@app.post('/predict')
async def inference(data: Census):

    model = load('model/model.joblib')
    encoder = load('model/encoder.joblib')
    lb = load('model/lb.joblib')

    data = data.dict(by_alias=True)
    data_frame = DataFrame(data, index=[0])
    columns = ["workclass",
               "education",
               "marital-status",
               "occupation",
               "relationship",
               "race",
               "sex",
               "native-country",
               "age",
               "hours-per-week",
               ]
    categorical_cols = columns[: -2]

    X, _, _, _ = data_package.process_data(
        data_frame,
        categorical_cols,
        encoder=encoder, lb=lb, training=False)
    pred = model_package.inference(model, X)
    y = lb.inverse_transform(pred)[0]
    return {"prediction": y}