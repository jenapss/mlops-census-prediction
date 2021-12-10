# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

## Intended Use
Identify features that affect person's income level

## Training Data
The training data was provided by UDACITY for Machine Learning DevOps Engineering learning and practice purposes and available onine. 
Extraction was done by Barry Becker from the 1994 Census database. A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))

Prediction task is to determine whether a person makes over 50K a year.
## Evaluation Data
Data is splitted into two subgroups such as ```train``` and ```test``` with ```train_test_split``` function with 0.2 test size.
## Metrics
Metrics that are mostly monitores are:
Presicion, Recall, and F1 score

## Ethical Considerations
Contains personal/private informations.
## Caveats and Recommendations
The dataset was collected in 1994 and does not reflect the today's reality and thus for real-world application for this project, the dataset needs to be updated wiht more fresh data. 
