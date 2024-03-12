#!/usr/bin/env python
# coding: utf-8

#  ## Data Cleaning

import pandas as pd
import numpy as np

def extract_column(data, col):
    data[col] = data[col].str.split(" ", expand=True)[0]

def convert_to_float(data, col):
    data[col] = pd.to_numeric(data[col])

def fill_missing_values(data, col):
    data[col].fillna(data[col].astype("float64").mean(), inplace=True)

def DataCleaning(data):
    #extract_column
    extract_column(data,'name')
    extract_column(data,'mileage')
    extract_column(data,'engine')
    extract_column(data,'max_power')
    # convert_to_float
    convert_to_float(data,'mileage')
    convert_to_float(data,'engine')
    convert_to_float(data,'max_power')
    #fill_missing_values
    fill_missing_values(data,'mileage')
    fill_missing_values(data,'engine')
    fill_missing_values(data,'seats')
    fill_missing_values(data,'max_power')
    
    data.dropna(subset=['fuel', 'transmission', 'owner'], inplace=True)
    data.drop(['torque','seller_type'],axis=1,inplace=True)
    data.drop(['mileage','seats'],axis=1,inplace=True) # D'apres le matrice corr
    data.replace('Manual',2, inplace = True)
    data.replace('Automatic',1, inplace = True)

    data.replace('First Owner',0, inplace = True)
    data.replace('Second Owner',1, inplace = True)
    data.replace('Third Owner & Above',2, inplace = True)

    data.replace('Diesel',1, inplace = True)
    data.replace('Petrol',2, inplace = True)
    data.replace('CNG',3, inplace = True)
    data.replace('LPG',4, inplace = True)

    #Changing types
    data['transmission'] = data['transmission'].astype(int)
    data.drop(columns=['name'], axis=1, inplace = True)
    return data




