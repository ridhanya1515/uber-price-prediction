import pickle
from flask import Flask, request, render_template
import numpy as np
import math

app = Flask(__name__)
model2 = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_feature = [int(i) for i in request.form.values()] #[1,2,3,5]

    final_feature = np.array(int_feature).reshape(1, -1)

    prediction = model2.predict(final_feature) #[[1,2,3,4]]
    output = round(prediction[0],2)

    return render_template('index.html', predict_text = 'Number of weekly rides {}'.format(math.floor(output)))

if __name__ == '__main__':
    app.run(debug=True)