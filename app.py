import os
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, escape, Response, flash, send_from_directory, Markup
from werkzeug.utils import secure_filename

from forms import AddForm

import numpy as np
import PIL
from PIL import Image
from keras.models import load_model
import dlib
from skimage import io
from test import *
from save import *

#comment
UPLOAD_FOLDER = './images'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# def triplet_loss(y_true, y_pred, alpha = 0.2):
#     anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
#     pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
#     neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
#     basic_loss = pos_dist - neg_dist + alpha
#     loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))    
#     return loss

FRmodel = load_model('./FRmodel.h5', custom_objects={'triplet_loss':triplet_loss})
graph = tf.get_default_graph()

@app.route('/')
def home():
    return render_template('welcome.html')

@app.route('/favicon.ico')
def fav():
    print('Logo')
    return render_template('welcome.html')

@app.route('/test')
def upload():
   return render_template('index.html')
	
@app.route('/tester', methods = ['GET', 'POST'])
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
        global graph
        with graph.as_default():
            possibles = identify(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(possibles)
            return render_template('results.html', possibles = possibles)


from pathlib import Path

my_file = Path('./info.pkl')
if my_file.is_file():
    with open('./info.pkl', 'rb') as pickle_file:
        info = pickle.load(pickle_file)
else:
    info = {}

my_file_1 = Path('./database.pkl')
if my_file_1.is_file():
    with open('./database.pkl', 'rb') as pickle_file:
        database = pickle.load(pickle_file)
else:
    database = {}

list1 = []

@app.route('/store', methods = ['GET', 'POST'])
def store():
    form = AddForm(request.form)
    if request.method == 'POST' and form.validate():
        childname = request.form['childname']
        parentname = request.form['parentname']
        email = request.form['email']
        phone = request.form['phone']
        address = request.form['address']
        key = childname + email
        if len(list1) == 0:
            list1.append(key)
        else:
            list1[0] = key
        info[key] = [childname, parentname, email, phone, address]
        f = open("./info.pkl","wb")
        pickle.dump(info,f)
        f.close()
        print(info)
        return redirect('/storeimage')
    return render_template('save.html', form = form)

@app.route('/storeimage')
def storeimage():
   return render_template('storeimage.html')

@app.route('/storeimager', methods = ['GET', 'POST'])
def storeimager():
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
        global graph
        with graph.as_default():
            embedding = process(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            database[list1[0]] = embedding
            f = open("./database.pkl","wb")
            pickle.dump(database,f)
            f.close()
            return redirect('/')


if __name__ == "__main__":
    app.run(host='0.0.0.0')