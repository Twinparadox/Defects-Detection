from keras import models, layers
from keras.applications import VGG16
from keras.applications.xception import Xception
from keras import Input
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint
import os
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math

# load model
model_path = './model'
model = load_model(os.path.join(model_path, "pretrained_VGG16_weight.hdf5"))
print("\n[INFO] Model loading success. Start testing... ")

# prediction
batch_size = 4
h = 200
w = 200
lr =1e-5
nb_classes = 2
transformation_ratio = 0.2
momentum = 0.9
test_list = './test.csv'
corr = 0
wron = 0

with open(test_list, 'r') as f:
    for l in f.readlines():
        img, cl = l.split(",")
        imgname = img
        cl = int(cl)
        img = Image.open(img).convert('RGB')
        img = img.resize((w ,h), Image.ANTIALIAS)
        img = np.array(img) / 255.
        img = np.expand_dims(img, 0)
        pred = (model.predict(img))[0].argmax()
        if pred == cl:
            corr += 1
            outp = "correct"
        else:
            wron += 1
            outp = "wrong"
        print("[INFO] Image {} prediction: {}. Label: {}, prediction: {}".format(imgname,
                                                                                 outp,
                                                                                 cl,
                                                                                 pred))
print("[INFO] Test accuracy: {}".format(float(corr ) /(corr +wron)))
