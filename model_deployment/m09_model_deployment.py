#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder


######## Precio de vehiculos

def predict_precios(Year, Mileage,State,Make,Model):

    modelo = joblib.load(os.path.dirname(__file__) + '/precios.pkl') 
    
    
    #Transformación de variables: 
     # Crear preprocesador
    num_features = ['Year', 'Mileage']
    cat_features = ['State', 'Make', 'Model']
    
    preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(), cat_features)
])   
        

    # Make prediction
    p1 = modelo.predict_precios

    return p1


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add an URL')
        
    else:

        url = sys.argv[1]

        p1 = predict_proba(url)
        
        print(url)
        print('Probability of Phishing: ', p1)
        
###### Fishing. 

def predict_proba(url):

    clf = joblib.load(os.path.dirname(__file__) + '/phishing_clf.pkl') 

    url_ = pd.DataFrame([url], columns=['url'])
  
    # Create features
    keywords = ['https', 'login', '.php', '.html', '@', 'sign']
    for keyword in keywords:
        url_['keyword_' + keyword] = url_.url.str.contains(keyword).astype(int)

    url_['lenght'] = url_.url.str.len() - 2
    domain = url_.url.str.split('/', expand=True).iloc[:, 2]
    url_['lenght_domain'] = domain.str.len()
    url_['isIP'] = (url_.url.str.replace('.', '') * 1).str.isnumeric().astype(int)
    url_['count_com'] = url_.url.str.count('com')

    # Make prediction
    p1 = clf.predict_proba(url_.drop('url', axis=1))[0,1]

    return p1


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add an URL')
        
    else:

        url = sys.argv[1]

        p1 = predict_proba(url)
        
        print(url)
        print('Probability of Phishing: ', p1)
        