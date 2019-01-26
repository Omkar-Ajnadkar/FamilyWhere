from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *

import h5py
import numpy as np
import PIL
from PIL import Image
from keras.models import load_model
import dlib
from skimage import io
import pickle

from flask import Flask, render_template

def triplet_loss(y_true, y_pred, alpha = 0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))    
    return loss

FRmodel = load_model('./FRmodel.h5', custom_objects={'triplet_loss':triplet_loss})
graph = tf.get_default_graph()

# global graph
# with graph.as_default():
from pathlib import Path

my_file_1 = Path('./database.pkl')
with open('./database.pkl', 'rb') as pickle_file:
    database = pickle.load(pickle_file)

with open('./info.pkl', 'rb') as pickle_file:
    info = pickle.load(pickle_file)
# database = pickle.load('./database.pkl')

def test_image(filename):
    image = io.imread(filename)
    face_detector = dlib.get_frontal_face_detector()
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(), x.right(), x.bottom()) for x in detected_faces]
    if len(face_frames) != 0:
        face_rect = face_frames[0]
        face = Image.fromarray(image).crop(face_rect)
        img = face.resize((96,96), Image.ANTIALIAS)
        img.save(filename)
    else:
        img = Image.open(filename)
        img = img.resize((96,96), Image.ANTIALIAS)
        img.save(filename)

def who_is_it(image_path, database, model):
    encoding = img_to_encoding(image_path, model)
    min_dist = 100
    
    distances = []
    for (name, db_enc) in database.items():
        dist = np.linalg.norm(encoding - db_enc)
        distances.append([dist, name])
        if dist < min_dist:
            min_dist = dist
            identity = name
    distances.sort()
    print(distances)
    possibles = {
        "first": [info[distances[0][1]][1], info[distances[0][1]][2],info[distances[0][1]][3]],
        "second": [info[distances[1][1]][1], info[distances[1][1]][2],info[distances[1][1]][3]],
        "third": [info[distances[2][1]][1], info[distances[2][1]][2],info[distances[2][1]][3]]
    }
    # print(possibles)
    return possibles
    # return render_template('results.html', possibles = possibles)    

def identify(filename):
    test_image(filename)
    possibles = who_is_it(filename, database, FRmodel)
    return possibles