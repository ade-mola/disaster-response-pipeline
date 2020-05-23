# Disaster Response Pipeline Project

## 1. Project Overview
In this project, I apply Data Engineering skills to analyze disaster data from [Figure Eight](https://www.figure-eight.com) to build an ETL and ML Pipeline and also build a model for an API that classifies disaster messages.
This project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

## 2. Project Components
There are three components in this project.

### a. ETL Pipeline
In a Python script, `data/process_data.py`, contains a data cleaning pipeline that:

   - Loads the `messages` and `categories` datasets
   - Merges the two datasets
   - Cleans the data
   - Stores it in a SQLite database
   
### b. ML Pipeline
In a Python script, `models/train_classifier.py`, contains a machine learning pipeline that:

   - Loads data from the SQLite database
   - Splits the dataset into training and test sets
   - Builds a text processing and machine learning pipeline
   - Trains and tunes a model using GridSearchCV
   - Outputs results on the test set
   - Exports the final model as a pickle file

### c. Flask Web App
In a Python script, `app/run.py`, contains web app that allows user to enter a query and see the resulting classifications.

## Instructions:
1. Run `pip install -r requirements.txt`

2. Run the following commands in the project's root directory to set up your database and model.

   - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
   - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
    `python app/run.py`

4. Go to http://0.0.0.0:3001/
