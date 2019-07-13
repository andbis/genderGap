#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 13:37:38 2018

@author: andreasbisgaard

Hardcoded pre-processing to explicitly show pre-processing steps
"""
from sklearn import preprocessing #preprocessing package
import numpy as np #package for matrix & vector manipulation
import pandas as pd #package to handle data in efficient dataframes

def start_preprocessing():
    #loading all the different datasets
    labels = pd.read_csv('data/labels.csv', delimiter=";")
    disc = pd.read_csv('data/discrimination.csv', delimiter=",")
    domestic = pd.read_csv('data/domesticviolence.csv', delimiter=",")
    abortion = pd.read_csv('data/legalabortion.csv', delimiter=",")
    legislation = pd.read_csv('data/legislation.csv', delimiter=",")
    proper = pd.read_csv('data/property.csv', delimiter=",")
    marriage = pd.read_csv('data/marriage.csv', delimiter=",")
    sex = pd.read_csv('data/sexualharassment.csv', delimiter=",")
    gdp = pd.read_csv('data/gdpcap.csv')#, index_col=0)#.iloc[:,-6]#, delimiter=",")
    gini = pd.read_csv('data/gini.csv')
    edu = pd.read_csv('data/edu.csv', delimiter=";")
    agb = pd.read_csv('data/agb.csv', delimiter=";")

    #PREPROCESSING
    #getting rid of countries not present in all sets
    #extracting all the unique countries of in the different sets to compare 
    discunique = disc.iloc[:,0].unique()
    labelsunique = labels.iloc[:,0].unique()
    uniquedomestic = domestic.iloc[:,0].unique()
    uniquelegislation = legislation.iloc[:,0].unique()
    uniqueproper = proper.iloc[:,0].unique()
    uniquemarriage = marriage.iloc[:,0].unique()
    uniquesex = sex.iloc[:,0].unique()
    uniquegdp = gdp.iloc[:,0].unique()
    uniquegini = gini.iloc[:,0].unique()
    uniqueedu = edu.iloc[:,0].unique()
    uniqueagb = agb.iloc[:,0].unique()

    #discarding examples in the discrimination data set that are not present in the label set
    discarddisc = []
    for i in discunique:
        if i not in labelsunique:
            discarddisc.append(i)
            disc.drop(disc[disc.Economy == i].index[0], inplace=True)
            
    discarddom = []
    for i in uniquedomestic:
        if i not in labelsunique:
            discarddom.append(i)
            domestic.drop(domestic[domestic.Economy == i].index[0], inplace=True)

    discardleg = []
    for i in uniquelegislation:
        if i not in labelsunique:
            discardleg.append(i)
            legislation.drop(legislation[legislation.Economy == i].index[0], inplace=True)
            
    discardprop = []
    for i in uniqueproper:
        if i not in labelsunique:
            discardprop.append(i)
            proper.drop(proper[proper.Economy == i].index[0], inplace=True)

    discardmar = []
    for i in uniquemarriage:
        if i not in labelsunique:
            discardmar.append(i)
            marriage.drop(marriage[marriage.Economy == i].index[0], inplace=True)
            
    discardsex = []
    for i in uniquesex:
        if i not in labelsunique:
            discardsex.append(i)
            sex.drop(sex[sex.Economy == i].index[0], inplace=True)

    countriessex = sex.iloc[:,0].unique()
    discardgdp = []
    for i in uniquegdp:
        if i not in countriessex:
            discardgdp.append(i)
            gdp.drop(gdp[gdp.iloc[:,0] == i].index[0], inplace=True)
            
    discardgini = []
    for i in uniquegini:
        if i not in countriessex:
            discardgini.append(i)
            gini.drop(gini[gini.iloc[:,0] == i].index[0], inplace=True)
            
    discardedu = []
    for i in uniqueedu:
        if i not in countriessex:
            discardedu.append(i)
            edu.drop(edu[edu.iloc[:,0] == i].index[0], inplace=True)
            
    discardagb = []
    for i in uniqueagb:
        if i not in countriessex:
            discardagb.append(i)
            agb.drop(agb[agb.iloc[:,0] == i].index[0], inplace=True)


    neunique = sex.iloc[:,0].unique()
    labeldel = []
    for i in labelsunique:
        if i not in neunique:
            labeldel.append(i)
            labels.drop(labels[labels.Country == i].index[0], inplace=True)

    uniqueabortion = abortion.iloc[:,0].unique()
    labelsunique = labels.iloc[:,0].unique()
    discardabor = []
    for i in uniqueabortion:
        if i not in labelsunique:
            discardabor.append(i)
            abortion.drop(abortion[abortion.COUNTRY == i].index[0], inplace=True)


    gdptrim = gdp.iloc[:,[0,-6]] # extracting the gdppercap 2013 columns, and the corresponding country
    ginitrim = gini.iloc[:,[0,-6]] # extracting the gini 2013 columns, and the corresponding country
    gdptrim.is_copy = False
    ginitrim.is_copy = False 
    #adding generic name "Economy" 
    gdptrim.rename(columns={'Country Name': 'Economy'}, inplace=True)
    ginitrim.rename(columns={'Country Name': 'Economy'}, inplace=True)
    abortion.rename(columns={'COUNTRY': 'Economy'}, inplace=True)

    #filling in nan values for gini
    means = {} #creating dictionary to hold mean gini value of 7 different regions 
    temp = list(sex.RegionName.unique()) #extracting the 7 different regions 

    for i in temp: #iterating over the the regions
        countries = list(sex[sex.iloc[:,1] == i].iloc[:,0]) #extracting all the countries that belong to this region
        col_gini = 0 #initiating int to hold cumulated gini coefficient for region
        k = 0 #counting the number of countries that contribute to the mean
        for country in countries: #iterating over the extracted countries
            gini = ginitrim[ginitrim.iloc[:,0] == country].iloc[0][1] #extracting gini for country

            if np.isnan(gini) == False: #if gini is not nan 
                k +=1 #increment k
                col_gini += gini #add to col_gini
        means[i] = col_gini/k #assign mean gini value for region to dictionary

    countries = ginitrim[np.isnan(ginitrim.iloc[:,1])].iloc[:,0] #extracting NaN Gini countries
    for i in countries:#iterating over the countries that has NaN values
        index = ginitrim[ginitrim.iloc[:,0] == i].index[0] #getting the index
        value = means[sex[sex.iloc[:,0] == i].iloc[:,1][sex[sex.iloc[:,0] == i].iloc[:,1].index[0]]] #getting value
        ginitrim.set_value(index, '2013', value) #inputting mean value to frame

    #mergin all sets into one dataframe
    main = pd.DataFrame.merge(disc, domestic, how='outer')
    main = pd.DataFrame.merge(main, legislation, how='outer')
    main = pd.DataFrame.merge(main, proper, how='outer')
    main = pd.DataFrame.merge(main, marriage, how='outer')
    main = pd.DataFrame.merge(main, sex, how='outer')
    main = pd.DataFrame.merge(main, abortion, how='outer')
    main = pd.DataFrame.merge(main, gdptrim, how='outer')
    main = pd.DataFrame.merge(main, ginitrim, on='Economy')
    main = pd.DataFrame.merge(main, edu, on='Economy')
    main = pd.DataFrame.merge(main, agb, on='Economy')
    #setting index of main frame to Economy
    main = main.set_index('Economy')

    #bulding a dataset of only women rights
    copy = main.iloc[:,:-4].copy() #temporary copy frame excluded the continuous values
    copy = copy.drop('RegionName', axis=1) #dropping the geographical feature RegionName from rights set
    rightsatt = list(copy.columns) #Extracting a list containing feature names, for plotting and investigation purposes 
    rights = np.zeros(copy.shape) #creating a numpy matrix to hold binary values 
    for idx, i in enumerate(copy.itertuples()): #iterating over copy frame
        for indel, item in enumerate(i): #iterating over columns 
            if indel != 0: #if it's not the first column, i.e. economy name
                if item == 'Yes': 
                    rights[idx, indel-1] = 1 #1 if yes to question
                elif item == 'No':
                    rights[idx, indel-1] = 0 #0 if no to question
                else:
                    rights[idx, indel-1] = np.nan   #nan if empty


    copy = main.copy() #new temp frame that contains all features 
    copy = copy.drop('RegionName', axis=1) #deleting RegionName feature
    x = np.zeros((119, copy.shape[1] + 7)) # creating a numpy matrix to contain all features plus regionnames
    allatt = list(copy.columns) #extracting the list of attributes 

    for i in list(main.RegionName.unique()): #appending the region names at the end of the attributes list
        allatt.append(i) 

    regions = list(main.RegionName.unique()) #extracting region names 

    for idx, i in enumerate(copy.itertuples()): #iterating over copy frame
        for indel, item in enumerate(i): #iterating over columns
            if indel == 0: #using Economy name to get region
                x[idx, 42 + regions.index(main.RegionName[item])] = 1 #dependent on the region's index in the region list
            elif indel == 39: #if the indel is GDP per cap
                if idx == 105: #if the country is 105: Syria 
                    x[idx, indel-1] = 35164000000 / 19810000 #manually inputting the GPDpercap for Syria
                else:
                    x[idx, indel-1] = item #otherwise input gdppercap for country
            elif indel == 40: #if the indel is gini 
                x[idx, indel-1] = item
            elif indel == 41: #if the indel is education
                x[idx, indel-1] = item
            elif indel == 42: #if indel is age at first birth
                x[idx, indel-1] = item
            
            else: #the feature is binary
                if item == 'Yes': 
                    x[idx, indel-1] = 1
                elif item == 'No':
                    x[idx, indel-1] = 0
                else:
                    x[idx, indel-1] = np.nan

    #changing name in attribute list
    allatt[allatt.index('2013_x')] = 'GDP per Cap' 
    allatt[allatt.index('2013_y')] = 'GINI'
    allatt[allatt.index('2013')] = 'Education'

    #deleting excess columns in labels data 
    del labels['ISO3']
    del labels['2013 Rank']
    labels = labels.set_index('Country') #setting the country as index
    labels = labels.sort_index() #sorting to correspond sequence in x and rights 


    #fitting an imputer, to replace NaNs with mode: a total of 30 countries have 15 or 16 nans 
    rightsprep = preprocessing.Imputer(axis=0, strategy="most_frequent").fit(rights)
    rights = rightsprep.transform(rights)

    #fitting an impute to replace NaNs with mode in complete set 
    xprep = preprocessing.Imputer(axis=0, strategy="most_frequent").fit(x)
    x = xprep.transform(x)
    return rights, rightsatt, x, allatt, labels, main
