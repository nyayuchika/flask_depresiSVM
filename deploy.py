from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
#load the model
model = pickle.load(open('depresimlmodel.sav', 'rb'))

@app.route('/')
def home():
    result = ''
    return render_template('./index.html', **locals())


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    text_in = str(request.form['text_in'])
    result = model.predict([[text_in]])[0]
    return render_template('./index.html', **locals())


if __name__ == '__main__':
    app.run(debug=True)