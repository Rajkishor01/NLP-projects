from transformers import pipeline
import numpy as np
import emoji
import re
import os

from flask import Flask,render_template,request

def preprocessing(comment):
    comment=emoji.demojize(comment,delimiters=(" "," "))
    comment=re.sub(r'http\S+|www\S+|https\S+',' ',comment)
    return comment

sentiment_classifier=pipeline('sentiment-analysis',model='cardiffnlp/twitter-roberta-base-sentiment') 

label_mapping = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

app=Flask(__name__)

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/analyze',methods=['POST'])
def analyze():
    comment_o=request.form['comment']
    comment=preprocessing(comment_o)
    
    sentiment=sentiment_classifier(comment)
    label=sentiment[0]['label']
    score=sentiment[0]['score']

    act_label=label_mapping[label]

    return render_template('result.html',comment=comment_o,sentiment=act_label,confidence=round(100*score,2))

if __name__=='__main__':
    app.run(debug=True)
