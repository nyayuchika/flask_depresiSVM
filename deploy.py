from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

app = Flask(__name__)
#load the model
model = pickle.load(open('depresimlmodel.sav', 'rb'))
vectorizer = pickle.load(open('vectorizer.sav', 'rb'))


# import os
# print(os.lisdtdir())

@app.route('/')
def home():
    result = ''
    return render_template('./index.html', **locals())


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    text_input = request.form.get('text_input','').strip()
    text_transformed = vectorizer.transform([text_input])
    #cek dulu gaiss
    print(text_transformed.toarray())

    
    result = model.predict(text_transformed)[0]
    return render_template('./index.html', **locals())


if __name__ == '__main__':
    app.run(debug=True)