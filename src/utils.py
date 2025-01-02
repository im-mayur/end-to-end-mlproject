import os 
import sys
import dill

import numpy as np 
import pandas as pd 
from src.exception import CustomException

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
    except Exception as e:
        raise CustomException(e, sys)
    
    except Exception as e:  
        raise CustomException(e, sys)
    


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report ={}
        print('length of models and params',len(models), len(params))
        for i, model_name in  enumerate(models.keys()):
            model = models[model_name]
            params = params.get(model_name, {})
            

            # model.fit(X_train, y_train)
            if params:
                gs = GridSearchCV(model, params, cv=3, n_jobs=-1,)
                gs.fit(X_train, y_train)

                model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred) 

            report[list(models.keys())[i]] = test_model_score  
        

        return report

    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file:
            return dill.load(file)
    except Exception as e:
        raise CustomException(e, sys)