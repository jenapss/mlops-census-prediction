from fastapi import FastAPI
from joblib import load
from pandas.core.frame import DataFrame
import starter.ml.model
from pydantic import BaseModel, Field
from starter import train_model, slice_score
from starter.ml import model as model_package
from starter.ml import data as data_package
import ruamel.yaml
import os
from census_class import Census

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


app = FastAPI()

@app.get('/')
async def get():
    return 'YAHOO CENSUS PREDICTION APP IS WORKING!'


@app.post('/predict')
async def inference(data: Census):

    model = load(os.path.join(os.getcwd(), 'model/model.joblib'))
    print('1')
    encoder = load(os.path.join(os.getcwd(), 'model/encoder.joblib'))
    lb = load(os.path.join(os.getcwd(), 'model/lb.joblib'))

    data = data.dict(by_alias=True)
    data_frame = DataFrame(data, index=[0])

    
    #load categories
    with open('config.yaml') as fp:
        data = ruamel.yaml.load(fp)
    columns = data['categorical_features']
    categorical_cols = columns[: -2]

    X, _, _, _ = data_package.process_data( data_frame, categorical_cols, encoder=encoder, lb=lb, training=False)
    pred = model_package.inference(model, X)
    y = lb.inverse_transform(pred)[0]
    print('PREDICTION---->',y)
    return {"PREDICTION": y}