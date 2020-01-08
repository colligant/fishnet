import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

from sys import stdout
from cv2 import resize, imwrite
from glob import glob
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, 
        UpSampling2D, Concatenate, Add, Flatten, Dense)
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.metrics import confusion_matrix
from random import shuffle

from data_generators import ClassificationDataGenerator


if __name__ == '__main__':

    n_classes = 4
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    last_conv = Conv2D(16, 3, name='conv_last', activation='relu')(base_model.output)
    flatten = Flatten()(last_conv)
    fc_1 = Dense(32, name='fc_1', activation='relu')(flatten)
    fc_out = Dense(n_classes, name='last_fc', activation='softmax')(fc_1)
    model = Model(inputs=base_model.input, outputs=fc_out)
    trainable = False
    for layer in model.layers:
        if layer.name == 'conv_last':
            trainable = True
        layer.trainable = trainable

    # model.summary()
    # size 270, 480, 256
    model_dir = 'models/class-models/'
    resize_image = (224, 224)
    train_path = '../classification-images/train/'
    test_path = '../classification-images/test/'
    train_generator = ClassificationDataGenerator(train_path, 8, resize=resize_image)
    test_generator = ClassificationDataGenerator(test_path, 8, resize=resize_image)
    tb = TensorBoard()
    model.compile(Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model_path = model_dir + 'test.h5'
    if not os.path.isfile(model_path):
        chpt = ModelCheckpoint(model_path, save_best_only=True, verbose=True, monitor='val_acc')
        model.fit_generator(train_generator,
                epochs=1000,
                callbacks=[chpt],
                validation_data=test_generator,
                verbose=1
                )
        model.save(model_dir + 'full.h5')
    else:
        model = tf.keras.models.load_model(model_path)
    test_generator = ClassificationDataGenerator(test_path, 1, resize=resize_image)
    for i, (image, m) in enumerate(test_generator):
        preds = np.squeeze(model.predict(image))
        fig, ax = plt.subplots()
        ax.imshow(image[0])
        plt.title(test_generator.index_to_class[np.argmax(m)])
        plt.show()
