#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jaspreet singh -ID:19150299

"""

import pandas as pd
import numpy as np
from selenium import webdriver
import requests
import bs4
from bs4 import BeautifulSoup
import time

"""
qualifying_results = pd.DataFrame()
for year in list(range(1994,2020)):
    url = 'https://www.formula1.com/en/results.html/{}/races.html'
    r = requests.get(url.format(year))
    soup = BeautifulSoup(r.text, 'html.parser')
    
    year_links = []
    for page in soup.find_all('a', attrs = {'class':"resultsarchive-filter-item-link FilterTrigger"}):
        link = page.get('href')
        if f'/en/results.html/{year}/races/' in link: 
            year_links.append(link)

    year_df = pd.DataFrame()
    new_url = 'https://www.formula1.com{}'
    for n, link in list(enumerate(year_links)):
        link = link.replace('race-result.html', 'starting-grid.html')
        df = pd.read_html(new_url.format(link))
        df = df[0]
        df['season'] = year
        df['round'] = n+1
        for col in df:
            if 'Unnamed' in col:
                df.drop(col, axis = 1, inplace = True)

        year_df = pd.concat([year_df, df])

    qualifying_results = pd.concat([qualifying_results, year_df])
    print("lol")
print(qualifying_results.shape)

qualifying_results.rename(columns = {'Pos': 'grid_position', 'Driver': 'driver_name', 'Car': 'car','Time': 'qualifying_time'}, inplace = True)
qualifying_results.drop('No', axis = 1, inplace = True)
qualifying_results.to_csv('qualifying.csv', index = False)
"""



races = pd.read_csv('Data/races.csv')

weather = races.iloc[:,[0,1,2]]
info = []

for link in races.url:
    print("lol")
    try:
        df = pd.read_html(link)[0]
        if 'Weather' in list(df.iloc[:,0]):
            n = list(df.iloc[:,0]).index('Weather')
            info.append(df.iloc[n,1])
        else:
            df = pd.read_html(link)[1]
            if 'Weather' in list(df.iloc[:,0]):
                n = list(df.iloc[:,0]).index('Weather')
                info.append(df.iloc[n,1])
            else:
                df = pd.read_html(link)[2]
                if 'Weather' in list(df.iloc[:,0]):
                    n = list(df.iloc[:,0]).index('Weather')
                    info.append(df.iloc[n,1])
                else:
                    df = pd.read_html(link)[3]
                    if 'Weather' in list(df.iloc[:,0]):
                        n = list(df.iloc[:,0]).index('Weather')
                        info.append(df.iloc[n,1])
                    else:
                        driver = webdriver.Chrome()
                        driver.get(link)

    except:
        info.append('not found')
        
        
len(info)
weather['weather'] = info

weather_dict = {'weather_warm': ['soleggiato', 'clear', 'warm', 'hot', 'sunny', 'fine', 'mild', 'sereno'],
               'weather_cold': ['cold', 'fresh', 'chilly', 'cool'],
               'weather_dry': ['dry', 'asciutto'],
               'weather_wet': ['showers', 'wet', 'rain', 'pioggia', 'damp', 'thunderstorms', 'rainy'],
               'weather_cloudy': ['overcast', 'nuvoloso', 'clouds', 'cloudy', 'grey', 'coperto']}
weather_df = pd.DataFrame(columns = weather_dict.keys())
for col in weather_df:
    weather_df[col] = weather['weather'].map(lambda x: 1 if any(i in weather_dict[col] for i in x.lower().split()) else 0)
    print("lol2")
    
weather_info = pd.concat([weather, weather_df], axis = 1)
weather_info.to_csv('weather.csv', index= False)





