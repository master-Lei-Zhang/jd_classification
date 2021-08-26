
import math
import numpy as np
import pandas as pd
from io import BytesIO
import os
import base64
from math import pi
import pickle
from bokeh.embed import components
from bokeh.models import ColumnDataSource, HoverTool, PrintfTickFormatter
from bokeh.plotting import figure
from bokeh.transform import factor_cmap, cumsum
from bokeh.models import BasicTicker, ColorBar, LinearColorMapper, PrintfTickFormatter
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.palettes import Category20c, Turbo256, RdYlBu
import requests
#import holoviews as hv
#hv.extension('bokeh')
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session, flash, jsonify
from flask_session import Session
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import matplotlib.pyplot as plt
plt.switch_backend('agg') 
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import metrics

filename_model = '../jd_classifier_model.sav'
model = pickle.load(open(filename_model, 'rb'))



def getPredictionResult(model, text):
    scores = model.predict_proba([text])[0]
    categories = ["da", "ds", "mle", "de", "other"]
    keywords = []
    if scores.max() < 0.5:
        prediction = [4]
    else:
        prediction = model.predict([text])
        word_indices = model['cv'].transform([text]).todense()[0]
        coef_ = model['lr'].coef_[prediction[0]]
        word_values = np.multiply(word_indices, coef_).tolist()[0]
        words = model['cv'].get_feature_names()
        weighted_words = sorted(list(zip(words, word_values)), key=lambda x: x[1], reverse=True)
        #print(weighted_words)
        for weighted_word in weighted_words[:10]:
            if weighted_word[1] <= 0:
                break
            keywords.append(weighted_word[0])
    if not keywords:
        prediction[0]=4
        cat_scores={'da': 0.25, 'ds': 0.25, 'mle': 0.25, 'de': 0.25}
    else:
        cat_scores = {categories[i]:scores[i] for i in range(len(scores))}
    
    return {"prediction": categories[prediction[0]], "keywords": keywords, "cat_scores": cat_scores}


def radar_chart_plot(cat_scores):
    num_vars = 4
    centre = 0.5

    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    # rotate theta such that the first axis is at the top
    theta += np.pi/2

    def unit_poly_verts(theta, centre ):
        """Return vertices of polygon for subplot axes.
        This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
        """
        x0, y0, r = [centre ] * 3
        verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
        return verts

    def radar_patch(r, theta, centre ):
        """ Returns the x and y coordinates corresponding to the magnitudes of 
        each variable displayed in the radar plot
        """
        # offset from centre of circle
        offset = 0.01
        yt = (r*centre + offset) * np.sin(theta) + centre 
        xt = (r*centre + offset) * np.cos(theta) + centre 
        return xt, yt
    verts = unit_poly_verts(theta, centre)
    x = [v[0] for v in verts] 
    y = [v[1] for v in verts] 

    p = figure(title="Baseline - Radar plot")
    text = ['DA','DS','MLE','DE','']
    source = ColumnDataSource({'x':x + [centre ],'y':y + [1],'text':text})

    p.line(x="x", y="y", source=source)

    labels = LabelSet(x="x",y="y",text="text",source=source)

    p.add_layout(labels)

    # example factor:
    f1 = np.array(cat_scores)
    flist = [f1]
    colors = ['blue','green','red', 'orange','purple']
    for i in range(len(flist)):
        
        xt, yt = radar_patch(flist[i], theta, centre)
        p.patch(x=xt, y=yt, fill_alpha=0.15, fill_color=colors[i])
    return p

app = Flask(__name__)
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)

@app.route('/', methods=['GET', 'POST'])
def Home():
    if request.method == 'POST':
        text = request.form['text']
    else:
        text = 'I want to be data scientist'
    #result= getPredictionResult(model, text)
    data = {'text':[text]}
    results = requests.post(request.url+'/api/v1/predict',json = data).json()
    cat_scores = results[0]['cat_scores']
    cats = ['da','ds','mle','de']
    scores = []
    for cat in cats:
        scores.append(cat_scores[cat])
    radar_chart = radar_chart_plot(scores)

    script_radar_chart, div_radar_chart = components(radar_chart)
    return render_template('index.html', div_radar_chart=div_radar_chart,script_radar_chart=script_radar_chart,text_display = text)




@app.route('/api/v1/predict', methods=['POST'])
def api_text():
    # Check if an ID was provided as part of the URL.
    # If ID is provided, assign it to a variable.
    # If no ID is provided, display an error in the browser.
    if 'text' in request.json:
        text = request.json['text']
    else:
        return "Error: No text field provided. Please specify a text."
    result = []
    for item in text:
        result.append(getPredictionResult(model, item))

    # Use the jsonify function from Flask to convert our list of
    # Python dictionaries to the JSON format.
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)


