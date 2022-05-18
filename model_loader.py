import tensorflow as tf
from keras.models import load_model

from flask import Flask
from flask import request

app = Flask(__name__)

@app.route('/predict',methods=['POST'])
def predict():
    model_name = request.json['modelName']
    data =  request.json['data']
    model = load_model("Models\\"+model_name+".h5")
    data_feauter = [data]
    pred_value = model.predict(data_feauter)
    if pred_value[0][0] >= 0.5 :
        return 'Yes'
    else :
        return 'No'


if __name__ == "__main__":
    app.run()