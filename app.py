import os
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, escape, Response, flash, send_from_directory, Markup
from werkzeug.utils import secure_filename

import numpy as np
import PIL
from PIL import Image
from keras.models import load_model
import dlib
from skimage import io
from test import *

#comment
UPLOAD_FOLDER = './images'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def triplet_loss(y_true, y_pred, alpha = 0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))    
    return loss

@app.route('/')
def home():
    return render_template('welcome.html')

@app.route('/favicon.ico')
def fav():
    print('Logo')
    return render_template('welcome.html')

@app.route('/upload')
def upload():
   return render_template('index.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        f = request.files['file']
        filename = secure_filename(f.filename)
        if filename == '':
            flash('No selected file')
            return redirect(request.url)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print(identify(os.path.join(app.config['UPLOAD_FOLDER'], filename)))
        return "Identified"
if __name__ == "__main__":
    app.run(host='0.0.0.0')
