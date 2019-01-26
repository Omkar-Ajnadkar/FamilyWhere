import numpy as np
import PIL
from PIL import Image
from keras.models import load_model
import dlib
from skimage import io
import pickle
import cv2

database = pickle.load('../python/database.pkl')
model = load_model('../python/FRmodel.h5')

def save_to_database(filename):
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

def img_to_encoding(filename, model):
    img1 = cv2.imread(filename, 1)
    img = img1[...,::-1]
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding

save_to_database(filename)
database[str(id)] = img_to_encoding(filename, FRmodel)

f = open("../python/database.pkl","wb")
pickle.dump(database,f)
f.close()