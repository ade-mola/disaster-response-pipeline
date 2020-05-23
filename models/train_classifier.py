"""
Disaster Resoponse Project
(Train Classifier)

Script Execution
> python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

INPUTS - 
    1) SQLite db path (containing pre-processed data)
    2) pickle file name to save ML model
"""

# Import Libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, make_scorer
import warnings
warnings.filterwarnings('ignore')

def load_data(database_filepath):
    
    """ Loads data from SQL Database.
    
        INPUTS - 
        database_filepath (str): SQL database filepath

        OUTPUTS - 
        X: Dataframe of features dataset
        Y: Dataframe of target labels dataset.
        category_names: List of target labels
    """
    # Load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse', con=engine)
    
    # Create X and Y datasets
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns

    return X, Y, category_names 


def tokenize(text):
    
    """ Tokenizes text data
    
        INPUT - 
        text (str): Messages for processing
        
        OUTPUT - 
        clean_words (list): Processed text after normalizing, tokenizing and lemmatizing
    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    words = word_tokenize(text)
    
    # remove stop words
    stopwords_ = stopwords.words("english")
    words = [word for word in words if word not in stopwords_]
    
    # extract root form of words
    clean_words = [WordNetLemmatizer().lemmatize(word, pos='v') for word in words]

    return clean_words


def build_model():
    
    """ Build model with GridSearchCV
    
        OUTPUT - 
        model: Trained model after performing grid search
    """
    # create pipeline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])

    # hyper-parameter grid
    parameters = {'clf__estimator__n_estimators': [10], 
                  'clf__estimator__min_samples_split': [2]}
    
    # create model
    model = GridSearchCV(estimator=pipeline,
            param_grid=parameters,
            verbose=3,
            cv=3)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    
    """ Shows model's performance on test data
    
        INPUTS - 
        model: trained model
        X_test: Test features
        Y_test: Test targets labels
        category_names: List of target labels
    """

    # predict
    y_pred = model.predict(X_test)

    # print classification report
    print(classification_report(Y_test.values, y_pred, target_names=category_names))

    # print accuracy score
    print('Accuracy: {}'.format(np.mean(Y_test.values == y_pred)))


def save_model(model, model_filepath):
    
    """ Saves the model to a Python pickle file 
    
        INPUT - 
        model: Trained model
        model_filepath: Filepath to save the model
    """

    # save model to pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()