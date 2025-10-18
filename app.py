import pickle
from flask import Flask, request, render_template
import numpy as np
import math
import os   

app = Flask(__name__)
model2 = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_feature = [int(i) for i in request.form.values()]  # [1,2,3,5]
    final_feature = np.array(int_feature).reshape(1, -1)
    prediction = model2.predict(final_feature)
    output = round(prediction[0], 2)
    return render_template('index.html', predict_text=f'Number of weekly rides {math.floor(output)}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
