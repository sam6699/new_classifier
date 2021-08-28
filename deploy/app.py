from flask import Flask, request
from flask import request
import joblib

tfidf = joblib.load("data/tfidf")
model = joblib.load("data/logistic_model")

app = Flask(__name__)


@app.route('/', methods=['GET','POST'])
def hello_world():
    html = '''
                <form method="POST">
                    <textarea name="text" placeholder="введите текст статьи"></textarea>
                    <input type="submit">
                </form>
        '''
    
    if request.method == "POST":
        html += model.predict(tfidf.transform([request.form['text']]))[0]

    return html
