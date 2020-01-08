from keras import models, layers
from keras.applications.xception import Xception
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam
from keras import Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math

epochs=300
batch_size = 4
h = 200
w = 200
lr =1e-4
nb_classes = 2
transformation_ratio = 0.2
momentum = 0.9

train_dir = os.path.join('./train')
validation_dir = os.path.join('./validation')
test_dir = os.path.join('./test')

classes = {}
print("[INFO] Classes indices: ")
print("=" * 40)
for idx, f in enumerate(sorted(os.listdir(train_dir))):
    classes[f] = idx
    print("%20s : %-20s" % (f, idx))
print("=" * 40)

train_datagen = ImageDataGenerator(rescale=1 / 255.,
                                   rotation_range=transformation_ratio * 100.0,
                                   shear_range=transformation_ratio,
                                   zoom_range=transformation_ratio,
                                   horizontal_flip=True,
                                   vertical_flip=True)
validation_datagen = ImageDataGenerator(rescale=1 / 255.)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=batch_size,
                                                    target_size=(h,w),
                                                    class_mode='categorical',
                                                    seed=42,
                                                    classes=classes)
validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              batch_size=batch_size,
                                                              target_size=(h,w),
                                                              class_mode='categorical',
                                                              seed=42,
                                                              classes=classes)

pretrained_xcept = Xception(input_shape=(h,w,3), weights=weights, include_top=False)
pretrained_xcept.trainable=True

additional_model = pretrained_xcept.output
additional_model = GlobalAveragePooling2D()(x)
additional_model = Dense(nb_classes, activation='softmax')(x)
model = Model(inputs=pretrained_xcept.input, outputs=additional_model)

checkpoint = ModelCheckpoint(filepath='pretrained_XCeption_weight.hdf5',
                             monitor='loss',
                             mode='min',
                             save_best_only=True)

additional_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['acc'])

history = additional_model.fit_generator(train_generator,
                                         steps_per_epoch=math.ceil(train_generator.n / train_generator.batch_size),
                                         epochs=epochs,
                                         validation_data=validation_generator,
                                         validation_steps=math.ceil(validation_generator.n / validation_generator.batch_size),
                                         callbacks=[checkpoint])

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Loss')
plt.legend()

plt.show()
