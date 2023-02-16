In this project, I applied the skills acquired in this course to develop a classification model on publicly available Census Bureau data. I also created unit tests to monitor the model performance on various data slices. Then,  deployed my model using the FastAPI package and created API tests. The slice validation and the API tests are incorporated into a CI/CD framework using GitHub Actions.


## Project Description

This project is a machine learning model that trains on the census.csv dataset to predict income based on various demographic factors. The dataset contains both categorical and numerical features and requires cleaning before being used for training.

The model is built using Python and uses FastAPI to create a RESTful API. The API allows users to make predictions using the trained model by sending a POST request with input data. The API also has a GET endpoint that returns a welcome message.

Continuous integration and deployment are set up using GitHub Actions and a cloud application platform, such as Heroku or Render. The project includes unit tests for both the model and the API, and a script to do a POST request on the live API for testing purposes.

Overall, this project demonstrates the full process of building and deploying a machine learning model with a RESTful API and continuous integration and deployment.
