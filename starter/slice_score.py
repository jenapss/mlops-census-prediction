import pandas as pd
from joblib import load
import ml.model
import logging






def slice_metrics():
    """
    Compute scores on data slices
    """
    test_data = pd.read_csv('data/clean_census_ts.csv')

    trained_model = load('model/model.joblib')
    encoder = load('model/encoder.joblib')
    lb = load('model/lb.joblib')

    slice_predictions = list()

    categories = list(test_data.select_dtypes(include='object').columns)[:-1]



    for cat_feature  in categories:
        for cls in test_data[cat_feature].unique():
            filtered_df = test_data[test_data[cat_feature] == cls]
            X_test, y_test, _, _ = process_data(filtered_df, label='salary', training=False, encoder=encoder, lb=binarizer)
            y_preds = trained_model.predict(X_test)
            precision, recall, fbeta = compute_model_metrics(y_test, y_preds)
            predictions[cat_feature] = {
                'precision': precision,
                'recall': recall,
                'fbeta': fbeta,
            }
            logging.info(predictions)
            slice_predictions.append(predictions)
    
    with open('data/slice_output.txt', 'w') as file:
        for slice_prediction in slice_predictions:
            file.write(slice_prediction + '\n')