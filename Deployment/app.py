import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
PICS_FOLDER = os.path.join('static', 'pics')
app.config['UPLOAD_FOLDER'] = PICS_FOLDER

def get_model():
    global model
    model = load_model('vgg16_adam_model.h5')
    print("Model loaded!")

def load_image(img_path):

    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    return img_tensor



def prediction(img_path):
    new_image = load_image(img_path)
    get_model()
    pred = model.predict(new_image)
    
    print(pred)
    
    y_prob=np.array(pred)
    y_label = y_prob.argmax(axis=-1)
    
    print(y_label)
    final=np.array(y_label)
    
    if final ==1:
        return "Forged!"
    else:
        return "Real."


# @app.route("/", methods=['GET', 'POST'])
# def home():


@app.route("/", methods = ['GET','POST'])
def main():
    if request.method == 'GET':
        return render_template('home.html')

    if request.method == 'POST':
        
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)                       #slashes should be handeled properly
        file.save(file_path)
        print(filename)
        product = prediction(file_path)
        print(product)
    return render_template('predict.html', product = product, user_image = file_path)


if __name__ == "__main__":
    app.run()