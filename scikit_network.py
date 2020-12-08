##  scikit_network.py:
##      Allows the AEDS software to make predictions about the user's
##      emotional state by comparing new vocal metrics against previously
##      stored metrics using the k-nearest neighbor algorithm.
import numpy as np
import pandas as pd
#from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from profileManager import *

## compare_new(): Compares data pertaining to the new recording and compares
##  it against the data retrieved from the user profile using the k-nearest
##  neighbor algorithm to find the closest match for the new data.
## Inputs: Data conveying the vocal metrics from the recent audio recording,
##          and the user profile.
## Outputs: The predicted emotional state of the user.
def compare_new(new_metrics, user_profile):
    # Changed the emotion data to use user profile data
    # Tim - 11/24
    emotion_data = user_profile.path
    df = pd.read_csv(emotion_data, header = None, sep = ',', names = ['Pitch', 'Tone', 'SPL', 'wordGap' , 'WordGapLen', 'Emotion', 'Score'])

    data = df.values
    y = df['Emotion']
    X = df[['Pitch', 'Tone', 'SPL' , 'wordGap' , 'WordGapLen']]
    
    # initialising model variable as knn classifier
    model = KNeighborsClassifier()
    #creating a dictionary for all the values we want to try for n_neighbors
    param_grid = {'n_neighbors': np.arange(1, 25)}
    #using gridsearch to test all values for n_neighbors
    knn_gscv = GridSearchCV(model, param_grid, cv=5)
    #fitting model to data
    knn_gscv.fit(X, y)

    #print("best parameter of n_neighbors",knn_gscv.best_params_)
    #print("best score of n_neighbors",knn_gscv.best_score_)
    
    new_metrics = new_metrics.reshape(1,-1)
    return knn_gscv.predict(new_metrics)

def compare_train(new_metrics, user_profile):
    # Changed the emotion data to use user profile data
    # Tim - 11/24
    emotion_data = user_profile.path
    df = pd.read_csv(emotion_data, header = None, sep = ',', names = ['Pitch', 'Tone', 'SPL', 'wordGap' , 'WordGapLen', 'Emotion', 'Score'])

    data = df.values
    y = df['Emotion']
    X = df[['Pitch', 'Tone', 'SPL' , 'wordGap' , 'WordGapLen']]
    
    # initialising model variable as knn classifier
    model = KNeighborsClassifier()
    #creating a dictionary for all the values we want to try for n_neighbors
    param_grid = {'n_neighbors': np.arange(1, 25)}
    #using gridsearch to test all values for n_neighbors
    knn_gscv = GridSearchCV(model, param_grid, cv=5)
    #fitting model to data
    knn_gscv.fit(X, y)
    
    new_metrics = new_metrics.reshape(1,-1)
    return knn_gscv.predict(new_metrics),knn_gscv.best_params_,knn_gscv.best_score_