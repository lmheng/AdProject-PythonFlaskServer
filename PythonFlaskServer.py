# import libraries
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open('PythonLogRegModel.pkl', 'rb'))


@app.route('/results', methods=['POST'])
def results():
    data = request.json
    print(type(data['ques']))
    print("data is", format(data))
    prediction = model.predict([np.array(data['ques'], dtype="int32")])
    output = prediction[0]
    return jsonify(output)


# run the server
if __name__ == '__main__':
    app.run(port=5000, debug=True)
