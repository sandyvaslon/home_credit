# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 23:19:02 2018

@author: S
"""
import pandas as pd
import numpy as np
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

rep=r'D:\Projets\Home_credit\data'
app_df=pd.read_csv(rep + r'\application_train.csv')
id=app_df['SK_ID_CURR'].tolist()
app_df=app_df.drop(['SK_ID_CURR'],axis=1)
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
#    else:
#        app_df[elem]=app_df[elem].convert_objects(convert_numeric=True)
        
#replace empty data with 0
app_df=app_df.replace(np.NaN, 0, regex=True)      
    
train, test = train_test_split(app_df, test_size=0.2)

# inputs   
y=train['TARGET']
X=train.drop(['TARGET'],axis=1)

#Model
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=120))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#training
model.fit(X, y, epochs=5, batch_size=32)

#Prediction
X=test.drop(['TARGET'],axis=1)
y=test['TARGET']
classes = model.predict(X)
proba=model.predict_proba(X)
print(classes[0:15])
loss_and_metrics = model.evaluate(X, y)
print(loss_and_metrics)