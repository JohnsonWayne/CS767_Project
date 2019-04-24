import os
from tensorflow.keras.models import model_from_json



def load_model():
    global model

    json_file = open('model.json', 'r')
    model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights("model.h5")
    model._make_predict_function()
    print("Model loaded")