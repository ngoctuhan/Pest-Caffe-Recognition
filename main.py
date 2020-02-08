from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf 
from PIL import Image
import numpy as np
import flask
import io
import cv2
import os
import pickle
import pandas as pd
import base64
import romdomName
from Detection import Model
from load_tf import CNN_Model

app = Flask(__name__)

model_dir = 'model'
md = CNN_Model(model_dir)

def load_Img():
    X_ = []
    df = pd.read_csv(r'dataImage.csv')
    for i in range (0, len(df['name'])):
        img = cv2.imread(os.path.join('compare', df['name'][i]))
        if img is not None:
            img = cv2.resize(img, (224, 224))
            X_.append(img)
    X_ = np.asanyarray(X_)
    return (df,X_)

def load_upload(filename):

    img = cv2.imread(os.path.join('Save', filename))
    if img is not None:
        img  = cv2.resize(img, (224,224))
    X = np.array([img])

    return X

@app.route('/predict', methods = ['GET', 'POST'])
def predict():


    if request.method == 'GET':
        return flask.jsonify('post a imgae')

    data = {"success": False}
    if request.method == "POST":

        res = request.get_json()
        if res is None:
            return flask.jsonify({'error': 'no file'})
        #print(res)
        imgString = res["file"]
        imgdata = base64.b64decode(imgString)
        filename = os.path.join('Save', romdomName.random_id(10))
        with open(filename, 'wb') as f:
            f.write(imgdata)

        img = cv2.imread( filename )
        img = cv2.resize(img, (224, 224))
        X = np.array([img])
        
        preds = md.predict_image(X)
        
        (df,X1) = load_Img()
        X1 = md.predict_image(X1)

        X1 = X1-preds
        X1 = X1**2
        X1 = np.sum(X1,axis = 1)
        sort = np.argsort(X1)

        result = sort[:10]

        data = {}
        data["result"] = []
        for i in result:
            if i>0:
                name = {'id' : str(df['id'][i+1]), 'name': str(df['image'][i+1]), 'dislay' : str(df['display_name'][i+1]) }
            else:
                name = {'id' : str(df['id'][i]), 'name': str(df['image'][i]) , 'dislay' : str(df['display_name'][i+1])}

            data["result"].append(name)
        
        return flask.jsonify(data)
        
    
if __name__ == "__main__":

    app.run('0.0.0.0', 5000 , debug=True)