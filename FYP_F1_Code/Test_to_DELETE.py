#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import svm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor


#importing libraries 

import seaborn as sns
import matplotlib.pyplot as plt 
from datetime import datetime
import math as ma
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter("ignore")


#importing other libraries for ML
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier

def score_classification(model):
    score = 0
    pred_df = pd.DataFrame()
    df_pred4 = pd.DataFrame()     
    for gp in df[df.season == 2020]['round'].unique():
        test = df[(df.season == 2020) & (df['round'] == gp)]
        X_test = test.drop(['finish_position','driver'], axis = 1)
        y_test = test.finish_position
        X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)

        # make predictions
        prediction_df = pd.DataFrame(model.predict_proba(X_test), columns = ['lose_prob', 'win_prob'])
        prediction_df['actual'] = y_test.reset_index(drop = True)
        prediction_df.sort_values('win_prob', ascending = False, inplace = True)
        prediction_df.reset_index(inplace = True, drop = True)
        prediction_df['predicted'] = prediction_df.index
        prediction_df['predicted'] = prediction_df.predicted.map(lambda x: 1 if x == 0 else 0)  
        
        prediction_df['round'] = gp
        prediction_df['season'] = 2020
        inversed = pd.DataFrame(scaler.inverse_transform(X_test))
        prediction_df['start_position'] = round(inversed[0])
        pred_df = pred_df.append(prediction_df, ignore_index = True)
        
        """print(prediction_df.head())"""
        score += precision_score(prediction_df.actual, prediction_df.predicted)

        df_pred = df[(df.season == 2020) & (df['round'] == gp)]
        df_pred2 = df_pred[['driver','round','start_position']].copy()
        df_pred3 = pd.merge(df_pred2, pred_df,on = ['round','start_position'])
        
        
    pred_df.to_csv('pred_df.csv')
    df_pred3.to_csv('prediction_df.csv')
    
    model_score = score / df[df.season == 2020]['round'].unique().max()
    print('The score of the model is: ')
    print(round(model_score*100))
   

df = pd.read_csv('final_df.csv', index_col=[0])
print(df.info())
df.finish_position = df.finish_position.map(lambda x: 1 if x == 1 else 0)
print(df.info())


train = df[df.season <2020]
X_train = train.drop(['finish_position','driver'], axis = 1)
y_train = train.finish_position


#ml models list
log_reg = LogisticRegression()
random_for_class = RandomForestClassifier()
dec_tree_class = DecisionTreeClassifier()


model_list = [dec_tree_class,random_for_class,log_reg]


scaler = [StandardScaler(),MinMaxScaler()]
for scaler in scaler:
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns) 
    print('Scores of every model by:',scaler)
    for model in model_list:
        model.fit(X_train, y_train)
        model_score = score_classification(model)  

print('Done')
