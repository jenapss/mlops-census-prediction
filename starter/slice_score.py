import pandas as pd
from joblib import load
import logging
import json
from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics

def slice_metrics():
    """
    Compute scores on data slices
    """
    test_data = pd.read_csv('data/clean_census_ts.csv')

    trained_model = load('model/model.joblib')
    encoder = load('model/encoder.joblib')
    lb = load('model/lb.joblib')

    slice_predictions = list()
    scores = dict()
    categories = list(test_data.select_dtypes(include='object').columns)[:-1]

    for cat_feature  in categories:
        for x in test_data[cat_feature].unique():
            filtered_df = test_data[test_data[cat_feature] == x]
            X_test, y_test, _, _ = process_data(filtered_df, categories, label='salary', training=False, encoder=encoder, lb=lb)
            y_preds = trained_model.predict(X_test)
            precision, recall, fbeta = compute_model_metrics(y_test, y_preds)

            scores[cat_feature] = {
                'precision': precision,
                'recall': recall,
                'fbeta': fbeta,
            }
            # logging.info(predictions)
            print(scores)
            #slice_predictions.append(scores)
    
    with open('data/slice_output.txt', 'w') as file:
        for key, value in scores.items(): 
            file.write('%s:%s\n' % (key, value))
        