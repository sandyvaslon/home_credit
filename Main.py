# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 23:19:02 2018

@author: S
"""
import pandas as pd
import numpy as np
#from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

rep=r'D:\Projets\Home_credit\data'
app_df=pd.read_csv(rep + r'\application_train.csv')

#dictionary for indexation
dico_main={}
for elem in app_df.columns:
    if not app_df[elem].dtype.kind in 'bif':
        dico={}
        for idx,mot in enumerate(app_df[elem].unique()):
            dico[mot]=idx
            #replacement
            app_df[elem]=app_df[elem].replace(mot,idx)
        dico_main[elem]=dico
    else:
        app_df[elem]=app_df[elem].convert_objects(convert_numeric=True)
    
train, test = train_test_split(app_df, test_size=0.2)
#    
y=train['TARGET']
X=train.drop(['TARGET'], axis=1)

del app_df

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)

X=test.drop(['TARGET'], axis=1)
pred=clf.predict(X)
