from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import json

import glob
import time

import os
import cv2
import numpy as np

import keras
from keras import backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

import base64
import io
from PIL import Image
from skimage.filters import gaussian

#** image utils ***************************************************************************
def sharpen(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5, multichannel=True)

    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)

def color_change(image, mask, color=[230, 50, 20]):
    b, g, r = color
    tar_color = np.zeros_like(image, dtype=np.uint8)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

    image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]

    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    changed = sharpen(changed)
    
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    imgout = image.copy()
    imgout[mask >0.5] = changed[mask >0.5]

    return imgout

def base64_to_image(b64Image):
    imgdata = base64.b64decode(b64Image)
    image = np.array(Image.open(io.BytesIO(imgdata)))
    return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

def image_to_base64(image):
    _, im_arr = cv2.imencode('.png', image)  
    im_bytes = im_arr.tobytes()    
    return 'data:image/png;base64,' + str(base64.b64encode(im_bytes))[2:][:-1]

#** Deep-Learning functions ***************************************************************************
def predict(image, height=224, width=224):
    im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    im = im / 255
    im = cv2.resize(im, (height, width))
    im = im.reshape((1,) + im.shape)
    
    pred = model.predict(im)
    
    mask = pred.reshape((224, 224))

    return mask

def setDevice(device = 0):
    os.environ["CUDA_VISIBLE_DEVICES"]= f"{device}"
    
    if not device == -1:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        config.log_device_placement = True  # to log device placement (on which device the operation ran)
        sess = tf.Session(config=config)
        set_session(sess)  # set this TensorFlow session as the default session for Keras 

#** Flask : REST Service ***************************************************************************
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def root():
    return 'welcome to flask'

@app.route('/infer', methods=['POST'])
def handle_post():
    data = request.get_data();
    params = json.loads(data, encoding='utf-8')
    if len(params) == 0:
        return jsonify({'result':'fail', 'msg': 'no parameter'})
    
    # load image
    image = base64_to_image(params['image'][22:])
    print('image : '+str(image.shape))
        
    # infer
    with graph.as_default():
        mask = predict(image)
    
    # color
    red = color_change(image, mask, [0, 0, 224])
    green = color_change(image, mask, [0, 224, 0])
    blue = color_change(image, mask, [224, 0, 0])
    yellow = color_change(image, mask, [0, 255, 223])
    cyan = color_change(image, mask, [255, 255, 0])
    pink = color_change(image, mask, [203, 192, 255])
    
    return jsonify({
        'result':'success'
        , 'red': image_to_base64(red)
        , 'green': image_to_base64(green)
        , 'blue': image_to_base64(blue)
        , 'yellow': image_to_base64(yellow)
        , 'cyan': image_to_base64(cyan)
        , 'pink': image_to_base64(pink)
    })

#** Main ***************************************************************************
if __name__ == '__main__':
    setDevice()
    model = keras.models.load_model('./models/hairnet_matting.hdf5')
    graph = tf.get_default_graph()
    
    app.debug = True
    app.run(host='0.0.0.0', port=5000)