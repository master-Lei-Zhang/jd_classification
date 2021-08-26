
import numpy as np
import os
import pickle
import requests
from flask import Flask, render_template, request, redirect, jsonify

filename_model = 'jd_classifier_model.sav'
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


app = Flask(__name__)
app.config.from_object(__name__)



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


