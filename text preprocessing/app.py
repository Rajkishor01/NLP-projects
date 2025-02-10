import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import Flask,render_template,request

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def text_preprocessing(text):

  # lower the word
  text=text.lower()

  # removing other special text
  text = re.sub(r'http\S+|www\S+', '', text)
  text = re.sub(r'<.*?>', '', text)
  text = re.sub(r'[^a-zA-Z\s]', '', text)

  # tokenization
  text=word_tokenize(text)

  # remove stopword
  stop_words=set(stopwords.words('english'))
  text=[word for word in text if word not in stop_words]

  # lemmatization
  lemmatizer=WordNetLemmatizer()
  text=[lemmatizer.lemmatize(word) for word in text]

  return ' '.join(text)


app=Flask(__name__)

@app.route('/')
def homepage():
  return render_template('index.html')


@app.route('/process',methods=['POST'])
def process():
  text=request.form['paragraph']
  processed_text=text_preprocessing(text)
  return render_template('result.html',original=text,processed=processed_text)

if __name__=='__main__':
  app.run(debug=True)