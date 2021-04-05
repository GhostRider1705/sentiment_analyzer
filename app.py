# !pip install flask-ngrok

from flask import Flask,request, url_for, redirect, render_template

from flask_ngrok import run_with_ngrok
import numpy as np
import pandas as pd
from Sentiment_predict import sentiment_predict
from Sentiment_predict import Tokenization
import pre_processing

from tensorflow.keras import models



TRAINED_MODEL_PATH_H5 = 'trained_model_h5'
EMBEDDING_MATRIX_PATH = 'main_embedding_matrix_pickle'

trained_model = models.load_model(TRAINED_MODEL_PATH_H5)
tokenizer = Tokenization(EMBEDDING_MATRIX_PATH)
# model=pickle.load(open('model.pkl','rb'))


app = Flask(__name__)
@app.route('/')
def hello_world():
    return render_template('sentiment.html')


@app.route('/predict',methods=['POST','GET'])
def predict():
    input_text=[x for x in request.form.values()]
    print(input_text)
    prediction,_= sentiment_predict(trained_model,[pre_processing.pre_processing(input_text[0])],tokenizer)
    # prediction = 10
    output=prediction

    if output>5:
        return render_template('sentiment.html',pred='Your Sentiment is Positive.\n Your score is {}'.format(output))
    elif output<5:
        return render_template('sentiment.html',pred='Your Sentiment is Negative.\n Your score is {}'.format(output))
    else:
        return render_template('sentiment.html',pred='Your Sentiment is Neutral.\n Your score is {}'.format(output))



app.run()
