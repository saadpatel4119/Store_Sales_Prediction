import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class AddColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['Item_Weight']=X['Item_Weight'].fillna(X['Item_Weight'].mode()[0])
        X['Outlet_Size']=X['Outlet_Size'].fillna(X['Outlet_Size'].mode()[0])
        values0 = ['Low Fat', 'low fat', 'LF']
        X['Item_Fat_Content']=['lowfat' if i in values0 else 'regular' for i in X['Item_Fat_Content']]
        X['Item_Fat_Content']=X['Item_Fat_Content'].astype('category')
        X['Outlet_Identifier']=[char.replace("OUT0","") for char in X['Outlet_Identifier']]
        X['Outlet_Identifier']=X['Outlet_Identifier'].astype(int)
        X['Outlet_Size']=X['Outlet_Size'].astype('category')
        X['Outlet_Location_Type']=[char.replace("Tier ","") for char in X['Outlet_Location_Type']]
        X['Outlet_Location_Type']=X['Outlet_Location_Type'].astype('int')
        X['Outlet_Type']=X['Outlet_Type'].astype('category')
        X['x']=[val[0:3] for val in X['Item_Identifier']]
        x=X['x'].value_counts().sort_values(ascending=False)
        X['x']=X['x'].map(x)
        X['x_1']=[val[3:] for val in X['Item_Identifier']]
        X=X.rename(columns={'x':'Item_Identifier_Part1','x_1':'Item_Identifier_Part2'})
        X['Item_Identifier_Part2']=X['Item_Identifier_Part2'].astype(int)
        X.drop(['Item_Identifier'],axis=1,inplace=True)
        X['Item_Fat_Content']=pd.get_dummies(X['Item_Fat_Content'],drop_first=True)
        # Calculate the frequency/count of each category
        encoding = X['Item_Type'].value_counts().to_dict()
        # Map the category to its frequency in a new column
        X['Item_Type'] = X['Item_Type'].map(encoding)
        X['Item_Type']=X['Item_Type'].astype('int')
        # Initialize the LabelEncoder
        label_encoder = LabelEncoder()

        # Fit and transform the categorical column
        X['Outlet_Size'] = label_encoder.fit_transform(X['Outlet_Size'])
        # Initialize the LabelEncoder
        label_encoder = LabelEncoder()

        # Fit and transform the categorical column
        X['Outlet_Type'] = label_encoder.fit_transform(X['Outlet_Type'])
        X['Item_Visibility'] = np.sqrt(X['Item_Visibility'])
        X['Item_Visibility']=X['Item_Visibility'].astype(float)
        X['Item_Visibility']=[0.25 if val==0 else val for val in X['Item_Visibility']]

        return X
  
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            columns = ['Item_Identifier',
                'Item_Weight',
                'Item_Fat_Content',
                'Item_Visibility',
                'Item_Type',
                'Item_MRP',
                'Outlet_Identifier',
                'Outlet_Establishment_Year',
                'Outlet_Size',
                'Outlet_Location_Type',
                'Outlet_Type']
            

            pipeline= Pipeline(
                steps=[
                ("preprocessing",AddColumnsTransformer()),
                ("scaler",StandardScaler())

                ]
            )

            logging.info(f"columns: {columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",pipeline,columns)
                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="Item_Outlet_Sales"

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)