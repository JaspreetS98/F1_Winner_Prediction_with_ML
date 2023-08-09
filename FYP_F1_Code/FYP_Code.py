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
import plotly.graph_objects as go
from plotly.offline import iplot
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
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix
from sklearn.metrics import classification_report
from sklearn import tree

from sklearn.tree import plot_tree, export_text

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV



def score_classification(model,df,scaler):
    score = 0
    pred_df = pd.DataFrame()
    df_pred4 = pd.DataFrame()     
    for gp in df[df.season == 2020]['round'].unique():
         #split test
        test = df[(df.season == 2020) & (df['round'] == gp)]
        X_test = test.drop(['finish_position','driver'], axis = 1)
        y_test = test.finish_position
        X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)
                
        #make predictions
        prediction_df = pd.DataFrame(model.predict(X_test), columns = ['pred_finish'])
        prediction_df['podium'] = y_test.reset_index(drop = True)
        prediction_df['actual'] = prediction_df.podium.map(lambda x: 1 if x == 1 else 0)
        prediction_df.sort_values('pred_finish', ascending = True, inplace = True)
        prediction_df.reset_index(inplace = True, drop = True)
        prediction_df['predicted'] = prediction_df.index
        prediction_df['predicted'] = prediction_df.predicted.map(lambda x: 1 if x == 0 else 0)
        
        #add useful columns to predictions
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
    print(final_df.columns)
    
    # changing columns labels to understand data easier 
    col_lab2 = {'grid':'start_position','positionOrder':'finish_position','rank':'fast_lap_rank',
                'year':'season','name_x':'gp','driverRef':'driver','nationality_x':'driver_nationality',
                'points_x':'points_driver_season','wins_x':'driver_season_wins','name_y':'constructor',
                'points_y':'constructor_season_points','position_x':'constructor_season_position',
                'wins_y':'constructor_season_wins','position_y':'qualify_position'}            
    final_df.rename(columns=col_lab2,inplace=True)  
    
    #check data frame info 
    print(final_df.info())
    print(final_df.shape)
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
    
    #drop drivers with an unrealistic age to race in the date of the GP 
    final_df.drop(final_df[final_df['age'] >= 60].index, inplace = True)
    
    #convert fast_lap_rank into numeric as it's not a string
    final_df['fast_lap_rank'] = pd.to_numeric(final_df['fast_lap_rank'],errors='coerce')
    
    #looking for null data and in %
    print(final_df.isnull().sum())
    print(final_df.isnull().sum() / len(final_df) * 100)

    #fill null values of fast lap rank and qualify_position with 20s as it's the last position on the grid  
    final_df['qualify_position'] = final_df['qualify_position'].fillna(20)
    final_df['fast_lap_rank'] = final_df['fast_lap_rank'].fillna(20)  
    
    #Change where start_position is 0 to finish_position to improve overtakes performance in model 
    final_df['start_position'] = np.where(final_df['start_position'] == 0, final_df['finish_position'], final_df['start_position'])

    #adding new column overtake to improve useful data
    overtakes = final_df['start_position'] - final_df['finish_position']
    final_df['overtakes'] = overtakes
    
    #mapping status column if finished or else
    final_df.status = final_df.status.map(lambda x: 1 if x == 'Finished' else 0)
    
    #check data frame info 
    print(final_df.info())
    print(final_df.shape)
    print(final_df.columns)
    print(final_df.isnull().sum())
    print(final_df.isnull().values.any())
    print(final_df.skew())

    #drop some columns
    final_df = final_df.drop(['raceId','statusId'],1)
    
    #DATA VISUALISATION
    
    #nationality of winning drivers
    fig = go.Figure(data=[go.Pie(labels=final_df[(final_df['finish_position']== 1)].sort_values(by=['driver_nationality'])['driver_nationality'].unique(),
                                                   values=final_df[(final_df['finish_position']== 1)].groupby('driver_nationality')['finish_position'].value_counts(),
                                                   hole=.3)])
    fig.show(renderer="svg")
    


    
    #show heatmap of data
    plt.figure(figsize=(10,8))
    sns.heatmap(final_df.corr(),annot=True, linewidths=2, linecolor='yellow',cmap="YlGnBu")
    plt.show()

    #drop some more columns
    final_df = final_df.drop(['points_driver_season','driver_season_wins','constructor_season_wins','constructor_season_points'],1)
    print(final_df.shape)
  
    #seperating categorical and numerical columns for understading 
    num_data = []
    cat_data = []
    for data in final_df.columns:
        if final_df[data].dtypes == 'O':
            cat_data.append(data)
        else:
            num_data.append(data)
    
    #label encoding categorical columns to use them into the ml model
    print(final_df.info())
    encoder = LabelEncoder()
    for i in cat_data:
        if i != 'driver':
            final_df[i] = encoder.fit_transform(final_df[i])   
    print(final_df.info())
    
    print(final_df.skew())
    print(num_data)
    print(cat_data)
    
    
    #MACHINE LEARNING 
    
    #split train
    train = final_df[final_df.season <2020]
    X_train = train.drop(['finish_position','driver'], axis = 1)
    y_train = train.finish_position
    
    #ml models list
    log_reg = LogisticRegression(solver = 'saga', penalty = 'l1', C = 100)
    random_for_class = RandomForestClassifier()
    dec_tree_class = DecisionTreeClassifier(max_leaf_nodes=1610, max_depth=17,criterion='gini',random_state=42)    
    model_list = [dec_tree_class,random_for_class,log_reg]

    """
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns) 
    
    #search best hyperparameteres for logistic regression
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    param = dict()
    param['solver'] = ['saga','liblinear']
    param['penalty'] = ['l1','l2,''none']
    param['C'] = [0.01,1,100]
    search = GridSearchCV(log_reg, param, scoring='accuracy', n_jobs=-1, cv=cv)
    result = search.fit(X_train, y_train)
    print('Higher Score: %s' % result.best_score_)
    print('Hyperparameteres to choose: %s' % result.best_params_)

    test = final_df[(final_df.season == 2020)]
    X_test = test.drop(['finish_position','driver'], axis = 1)
    y_test = test.finish_position
    X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)
   
    #search best hyperparameteres for decision tree
    dec_tree_class.fit(X_train, y_train)
    
    for max_l in range(1600,1620):
      model = DecisionTreeClassifier(max_leaf_nodes=max_l,random_state=42)
      model.fit(X_train, y_train)
      print('The Training Accuracy for max_depth {} is:'.format(max_l), model.score(X_train, y_train))
      print('The Validation Accuracy for max_depth {} is:'.format(max_l), model.score(X_test,y_test))
      print('')
    for max_d in range(2,20):
      model = DecisionTreeClassifier(max_depth=max_d,random_state=42)
      model.fit(X_train, y_train)
      print('The Training Accuracy for max_depth {} is:'.format(max_d), model.score(X_train, y_train))
      print('The Validation Accuracy for max_depth {} is:'.format(max_d), model.score(X_test,y_test))
      print('')
    print(dec_tree_class.score(X_train, y_train))
    print(dec_tree_class.score(X_test, y_test))
    """     

    stds = StandardScaler()
    mms = MinMaxScaler()
    scaler = [stds,mms]
    for scaler in scaler:
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns) 
        print('Scores of every model by:',scaler)
        for model in model_list:
            model.fit(X_train, y_train)
            score_classification(model,final_df,scaler)  

    
    

if __name__ == "__main__":
    main()

