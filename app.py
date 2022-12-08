# -*- coding: utf-8 -*-
"""
Created on Mon May 16 12:36:23 2022

@author: YASHIM GABRIEL
"""

import numpy as np 
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('RandomForest-97b.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    ag = int(request.form['age'])
    nsp = int(request.form['nsp'])
    fsi = int(request.form['fsi'])
    NoP = int(request.form['NoP'])
    smoke = int(request.form['smoke'])
    smokeYr = int(request.form['smokeYr'])
    smokePYr = int(request.form['smokePYr'])
    HoCon = int(request.form['HoCon'])
    HoConYr = int(request.form['HoConYr'])
    iud = int(request.form['iud'])
    iudYr = int(request.form['iudYr'])
    std = int(request.form['std'])
    stdGH = int(request.form['stdGH'])
    stdND = int(request.form['stdND'])
    stdTFD = int(request.form['stdTFD'])
    Dx = int(request.form['Dx'])
    Hin = int(request.form['Hin'])
    Sch = int(request.form['Sch'])
    Cit = int(request.form['Cit'])
    name = request.form['name']


    final_features = np.array([[ag,nsp,fsi,NoP,smoke,smokeYr,smokePYr,HoCon,HoConYr,iud,iudYr,std,stdGH,stdND,stdTFD,Dx,Hin,Sch,Cit]])
    prediction = model.predict(final_features)

    output = prediction[0]
    
    if output == 0:
        result = 'You do not have Cervical Cancer'
    else:
        result = 'You have Cervical Cancer'
    
    return render_template('results.html', name_text= name, prediction_text=result)
     

if __name__ == "__main__":
    app.run(debug=True)
