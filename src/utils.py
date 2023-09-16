import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def Item_Visibility_Preprocessing(df):
    df['Item_Visibility'] = np.sqrt(df['Item_Visibility'])
    df['Item_Visibility']=df['Item_Visibility'].astype(float)
    df['Item_Visibility']=[0.25 if val==0 else val for val in df['Item_Visibility']]

def Item_Identifier_Preprocessing(df):
    df['x']=[val[0:3] for val in df['Item_Identifier']]
    x=df['x'].value_counts().sort_values(ascending=False)
    df['x']=df['x'].map(x)
    df['x_1']=[val[3:] for val in df['Item_Identifier']]
    df=df.rename(columns={'x':'Item_Identifier_Part1','x_1':'Item_Identifier_Part2'})
    df['Item_Identifier_Part2']=df['Item_Identifier_Part2'].astype(int)
    df.drop(['Item_Identifier'],axis=1,inplace=True)

def Item_Fat_Content_Preprocessing(df):
    values0 = ['Low Fat', 'low fat', 'LF']
    df['Item_Fat_Content']=['lowfat' if i in values0 else 'regular' for i in df['Item_Fat_Content']]
    df['Item_Fat_Content']=df['Item_Fat_Content'].astype('category')


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

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
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)