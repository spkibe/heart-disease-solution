#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:51:25 2020

@author: spkibe
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('modle.pkl', 'rb'))

@pp.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering result on HTMl GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.prediction(final_features)
    
    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text='Patient diagnosis is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)