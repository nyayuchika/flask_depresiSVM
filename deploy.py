from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize

app = Flask(__name__)
#load the model
model = pickle.load(open('new_depresimlmodel.sav', 'rb'))
vectorizer = pickle.load(open('new_vectorizer.sav', 'rb'))

# import os
# print(os.lisdtdir())

@app.route('/')
def home():
    result = ''
    return render_template('./index.html', **locals())


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    #ngambil input
    text_input = request.form.get('text_input','').strip()
    
    #lowercase
    lowercase = text_input.lower()
   
    #remove punct
    import re
    punctuation = re.sub("[^\w\s\d]","",lowercase)
   
    #convert slang
    alay_dict = pd.read_csv('new_kamusalay.csv', encoding='latin-1', header=None)
    alay_dict = alay_dict.rename(columns={0:'original', 1:'replacement'})
    alay_dict_map = dict(zip(alay_dict['original'], alay_dict['replacement']))
    def normalize_alay(text):
        return " ".join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split()])
    normalize_alay = normalize_alay(punctuation)
   
    #remove stopwords
    nltk.download('punk')
    nltk.download('stopwords')
    text_tokens = word_tokenize(normalize_alay)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    filtered_sentence = (" ").join(tokens_without_sw)
    filtered_sentence = [filtered_sentence]

    #convert input
    text_transformed = vectorizer.transform(filtered_sentence)
    # #cek dulu gaiss
    # array = text_transformed.toarray()

    #result
    result = model.predict(text_transformed)[0]
    # confidence_score = model.decision_function(text_transformed)
    
    return render_template('./index.html', **locals())
    


if __name__ == '__main__':
    app.run(debug=True)