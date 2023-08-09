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
 
def class_model(model,final_df,std_scal):
    score = 0
    for gp in final_df[final_df.season == 2018]['round'].unique():
        test = final_df[(final_df.season == 2018) & (final_df['round'] == gp)]
        X_test = test.drop(['finish_position','driver'], axis = 1)
        y_test = test.finish_position
        X_test = pd.DataFrame(std_scal.transform(X_test), columns = X_test.columns)
    
        # make predictions
        pred_df = pd.DataFrame(model.predict_proba(X_test), columns = ['lose_prob', 'win_prob'])
        pred_df['actual'] = y_test.reset_index(drop = True)
        pred_df.sort_values('win_prob', ascending = False, inplace = True)
        pred_df.reset_index(inplace = True, drop = True)     
        pred_df['predicted'] = pred_df.index
        pred_df['predicted'] = pred_df.predicted.map(lambda x: 1 if x == 0 else 0)

        print(pred_df.head())
        score += precision_score(pred_df.actual, pred_df.predicted)
    
    pred_df.to_csv('prediction_df.csv')
    model_score = score / final_df[final_df.season == 2018]['round'].unique().max()
    print('The score of the model is: ')
    print(round(model_score*100))


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
    df1 = pd.merge(results_df, races_df, how='inner', on ='raceId').drop(['url','time_y',
                  'time_x','fastestLap','circuitId','number','position','positionText',
                  'fastestLapTime','fastestLapSpeed','points','milliseconds'], axis = 1) 
    df2 = pd.merge(df1, drivers_df, on = 'driverId').drop(['number','code','forename','surname','url'],1)
    df3 = pd.merge(df2, driver_standings_df, on = ['driverId', 'raceId']).drop(['positionText','position'],1)
    df4 = pd.merge(df3, constructors_df, on = ['constructorId']).drop(['constructorRef','url','nationality_y'],1)
    df5 = pd.merge(df4, constructor_standings_df, on = ['constructorId', 'raceId']).drop(['positionText'],1)
    df6 = pd.merge(df5, qualifying_df, how='left', on = ['driverId', 'raceId']).drop(['q1','q2','q3','number',
                                                                                      'constructorId_y'], axis = 1)
    final_df = pd.merge(df6, status_df, on ='statusId')
    final_df = final_df.drop(['resultId','driverId','driverStandingsId','constructorStandingsId',
                              'qualifyId','constructorId_x'],1)
    #check data frame info 
    print(final_df.info())
    print(final_df.shape)
    print(final_df.columns)
    
    # changing columns labels to understand data easier 
    col_lab2 = {'grid':'start_position','positionOrder':'finish_position','rank':'fast_lap_rank',
                'year':'season','name_x':'gp','driverRef':'driver','nationality_x':'driver_nationality',
                'points_x':'points_driver_season','wins_x':'driver_season_wins','name_y':'constructor',
                'points_y':'constructor_season_points','position_x':'constructor_season_position',
                'wins_y':'constructor_season_wins','position_y':'qualify_position'}            
    final_df.rename(columns=col_lab2,inplace=True)
    print(final_df.columns)
    
    #change string to date format
    pd.to_datetime(final_df.date)
    final_df['dob'] = pd.to_datetime(final_df['dob'])
    final_df['date'] = pd.to_datetime(final_df['date'])
    
    #get driver's age at every race and insert it in a new column then adjust data frame
    difference = final_df['date']-final_df['dob']
    age = difference.dt.days/365
    final_df['age'] = round(age)
    final_df = final_df.drop(['dob','date'], axis = 1)
    
    #looking for null data and in %
    print(final_df.isnull().sum())
    print(final_df.isnull().sum() / len(final_df) * 100)

    #fill null values of fast lap rank with 0s  
    final_df['qualify_position'] = final_df['qualify_position'].fillna(0)
    final_df['fast_lap_rank'] = final_df['fast_lap_rank'].fillna(0)

    #looking for null data after operations
    print(final_df.isnull().sum())
    print(final_df.shape)    

    #adding new column overtake to improve useful data
    overtakes = final_df['start_position'] - final_df['finish_position']
    final_df['overtakes'] = overtakes
    
    #checking missing data
    print(final_df.isna().sum())
    print(final_df.info())
    
    #changing datatype and fill empty
    final_df['fast_lap_rank'] = pd.to_numeric(final_df['fast_lap_rank'],errors='coerce')
    final_df['fast_lap_rank'] = final_df['fast_lap_rank'].fillna(0)
    
    #create new csv for data vis
    final_df.to_csv('final_df_for_data_vis.csv')

    #drop some columns
    final_df = final_df.drop(['raceId','status','constructor_season_wins','constructor_season_points','raceId'],1)
    print(final_df.shape)
    
    #show heatmap of data
    plt.figure(figsize=(10,8))
    sns.heatmap(final_df.corr(),annot=True, linewidths=2, linecolor='yellow',cmap="YlGnBu")
    plt.show()
    
    #get dummies
    final_df = pd.get_dummies(final_df, columns = ['driver_nationality', 'constructor','gp'])
    print(final_df.shape)    
    
    
    """
    print(final_df.info())
    final_df.to_csv('final_df.csv')
    """
    final_df.finish_position = final_df.finish_position.map(lambda x: 1 if x == 1 else 0)
    print(final_df.info())

    train = final_df[final_df.season <2018]
    X_train = train.drop(['driver', 'finish_position'], axis = 1)
    y_train = train.finish_position

    #ml models list
    log_reg = LogisticRegression()
    random_for_class = RandomForestClassifier()
    dec_tree_class = DecisionTreeClassifier()
    model_list = [dec_tree_class,random_for_class,log_reg]
    
    std_scal = StandardScaler()
    X_train = pd.DataFrame(std_scal.fit_transform(X_train), columns = X_train.columns) 
    print('Scores of every model by',std_scal)
    for model in model_list:
        model.fit(X_train, y_train)
        class_model(model,final_df,std_scal)   


if __name__ == "__main__":
    main()

