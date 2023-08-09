#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jaspreet singh
"""

#importing libraries 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn import tree

def standard_scaler(train_x,test_x,model_list,train_y,test_y):
    # fit stanard scaler on training data
    std = StandardScaler().fit(train_x)
    #transform testing and training data
    train_x_std_scal = std.transform(train_x)
    test_x_std_scal = std.transform(test_x)
    models_loop(train_x,test_x,model_list,train_y,test_y,train_x_std_scal,test_x_std_scal)

def min_max_scaler(train_x,test_x,model_list,train_y,test_y):
    #fit min max scaler on training data
    min_max_scal = MinMaxScaler().fit(train_x)
    #transform testing and training data
    train_x_mms = min_max_scal.transform(train_x)
    test_x_mms = min_max_scal.transform(test_x)
    models_loop(train_x,test_x,model_list,train_y,test_y,train_x_mms,test_x_mms)

def models_loop(train_x,test_x,model_list,train_y,test_y,train_x_2,test_x_2):
    result = {}
    for models in model_list:
        models.fit(train_x_2,train_y)
        prediction_y = models.predict(test_x_2)
        print(models,":",accuracy_score(prediction_y,test_y)*100)
        result.update({str(models):models.score(test_x_2,test_y)*100})
        """
        if models == 'dec_tree_class' or 'random_for_class':
            #get feature importance
            importance = models.feature_importances_
            for a,b in enumerate(importance):
                print('Feature: %0d, Score: %.5f' % (a,b))
        """
 
def main():

    #getting data from csv into data frames
    status_df = pd.read_csv('Data/status.csv')
    qualifying_df = pd.read_csv('Data/qualifying.csv')
    drivers_df = pd.read_csv('Data/drivers.csv')
    driver_standings_df = pd.read_csv('Data/driver_standings.csv')
    races_df = pd.read_csv('Data/races.csv')
    results_df = pd.read_csv('Data/results.csv')
    constructor_standings_df = pd.read_csv('Data/constructor_standings.csv')
    constructors_df = pd.read_csv('Data/constructors.csv')
    circuits_df = pd.read_csv('Data/circuits.csv')
    
    #print all data frame columns
    pd.set_option('display.max_columns', None)
    
    #merging all data frames into one
    df1 = pd.merge(results_df, races_df, on ='raceId').drop(['url'], axis = 1)
    df2 = pd.merge(df1, drivers_df, on = 'driverId')
    df3 = pd.merge(df2, driver_standings_df, on = ['driverId', 'raceId'])
    df4 = pd.merge(df3, constructors_df, on = ['constructorId'])
    df5 = pd.merge(df4, constructor_standings_df, on = ['constructorId', 'raceId'])
    df6 = pd.merge(df5, qualifying_df, on = ['driverId', 'raceId']).drop(['q1','q2','q3','number'], axis = 1)
    final_df = pd.merge(df6, status_df, on ='statusId')
    print(final_df.shape)
    #check data frame info 
    print(final_df.columns)
    print(final_df.head())

    #reduce volume of missing data to make it more closer to reality
    """final_df = final_df[final_df['status'] == 'Finished']"""

    
    # droping columns not needed
    final_df = final_df.drop(['url_x', 'url_y', 'constructorRef', 'nationality_y','driverId',
                              'number_y', 'driverRef' ,'forename' , 'surname','time_y',
                              'positionText_x','number_x','positionOrder','fastestLapTime',
                              'time_x','fastestLap','positionText_y','statusId','positionText',
                              'circuitId','constructorId_y','qualifyId','driverStandingsId','status'],1)
    
    # changing columns labels to understand data easier 
    col_lab2 = {'grid':'start_position','position_x':'finish_position','fastestLapSpeed':'fastest_speed',
     'name_x':'gp','code':'driver_code','nationality_x':'driver_nationality','rank':'fast_lap_rank',
     'points_x':'points_race','name_y':'constructor','year':'season','position':'qualify_position',
     'points_y':'constructor_points','position_y':'constructor_position','wins_y':'constructor_wins',
     'wins_x':'driver_wins'}
    final_df.rename(columns=col_lab2,inplace=True)
    print(final_df.columns)
    print(final_df.head())
    

    #change string to date format
    pd.to_datetime(final_df.date)
    final_df['dob'] = pd.to_datetime(final_df['dob'])
    final_df['date'] = pd.to_datetime(final_df['date'])
    
    #get driver's age at every race and insert it in a new column then adjust data frame
    difference = final_df['date']-final_df['dob']
    age = difference.dt.days/365
    final_df['age'] = round(age)
    final_df = final_df.drop(['dob','date'], axis = 1)
    
    #checking and changing datatype
    print(final_df.isna().sum())
    print(final_df.info())
    print(final_df.columns)
    print('lol')
    for cols in final_df.columns:
        print(final_df[cols].unique())
    
    
    
    data_to_change = ['milliseconds','fast_lap_rank','fastest_speed','finish_position']
    for data in data_to_change:
        final_df[data] = pd.to_numeric(final_df[data],errors='coerce')
    print(final_df.info())

    #looking for null data and in %
    print(final_df.isnull().sum())
    print(final_df.isnull().sum() / len(final_df) * 100)
   
    
    #put avarage fastest speed and milliseconds for null values
    final_df['milliseconds'] = final_df['milliseconds'].fillna(final_df['milliseconds'].mean())
    final_df['fastest_speed']= final_df['fastest_speed'].fillna(final_df['fastest_speed'].mean())
    #fill null values of fast lap rank with 0s
    final_df[['fast_lap_rank']] = final_df[['fast_lap_rank']].fillna(0)
    #looking for null data after operations
    print(final_df.isnull().sum())
    print(final_df.shape)
    print('lol')

       
    #drop some columns
    final_df = final_df.drop(['resultId', 'raceId', 'constructorId_x','milliseconds'],1)
    print(final_df.shape)

    

    # seperating categorical and numerical columns for understading 
    num_data = []
    cat_data = []
    for data in final_df.columns:
        if final_df[data].dtypes == 'O':
            cat_data.append(data)
        else:
            num_data.append(data)
            
    #check numerical and categorical data
    print(final_df[cat_data].head())
    print(final_df[num_data].head())
    
    #check skewness of the data
    print(final_df.skew())
    
    #removing the outliners using the quantiles
    Q1 = final_df.quantile(0.25)
    Q3 = final_df.quantile(0.75)
    IQR = Q3 - Q1
    final_df = final_df[~((final_df<(Q1-1.5*IQR)) | (final_df>(Q3+1.5*IQR))).any(axis=1)]
    print(final_df.skew())
    
    #show heatmap of data
    plt.figure(figsize=(10,8))
    sns.heatmap(final_df.corr(),annot=True, linewidths=2, linecolor='yellow',cmap="YlGnBu")
    plt.show()
    
    #label encoding categorical columns to use them into the ml model
    print(final_df.info())
    encoder = LabelEncoder()
    for i in cat_data:
        final_df[i] = encoder.fit_transform(final_df[i])   
    print(final_df.info())
    
    
    #drop some columns and merge 
    final_dummies = final_dummies.drop(['on_podium','first'],1)
    final_df = pd.merge(final_df, final_dummies, on =['start_position','finish_position',
                                                      'points_race','laps','fast_lap_rank',
                                                      'fastest_speed','season','round','wins',
                                                      'age','overtakes','pace','qualify_position'])    
    final_df = final_df.drop(['gp','driver_nationality','constructor','driver_code_x'],1)
    col_lab2 = {'driver_code_y':'driver'}
    final_df.rename(columns=col_lab2,inplace=True)
    final_df.to_csv('final_df.csv')
    
    print(final_df.skew())
    
    #ml models list
    log_reg = LogisticRegression()
    random_for_class = RandomForestClassifier()
    gauss_nb = GaussianNB()
    dec_tree_class = DecisionTreeClassifier()
    sgd_class = SGDClassifier()
    knn_class = KNeighborsClassifier()
    model_list = [dec_tree_class,random_for_class,log_reg,gauss_nb,sgd_class,knn_class]
    
    #split data 
    x = final_df.drop('driver_code',1)
    y = final_df.driver_code
    
    #split test and train data
    train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.2,random_state=99)
    
    
    #call functions to execute training and testing loops
    print("Here the base resuts of the models: \n")
    models_loop(train_x,test_x,model_list,train_y,test_y,train_x,test_x)
    print("\n\n")
    print("Here the resuts of the models with standard scaler: \n")
    min_max_scaler(train_x,test_x,model_list,train_y,test_y)
    print("\n\n")
    print("Here the resuts of the models with min max scaler: \n")
    standard_scaler(train_x,test_x,model_list,train_y,test_y)

    
if __name__ == "__main__":
    main()

