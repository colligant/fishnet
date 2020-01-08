import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import matplotlib.pyplot as plt
import numpy as np

from sys import stdout
from cv2 import resize, imwrite
from glob import glob
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, 
        UpSampling2D, Concatenate, Add)
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.metrics import confusion_matrix
from random import shuffle

from data_generators import DataGenerator
from train_model import simple_fcnn, precision_and_recall_from_generator

def label_and_save_images_with_model(model, files, out_path):
    out_dir_mask = os.path.join(out_path, 'masks/')
    out_dir_image = (os.path.join(out_path, '/images/'))
    for j, i in enumerate(files):
        if not os.path.isfile(os.path.splitext(i)[0] + '__labels.json'):
            stdout.write('{}/{}\r'.format(j+1, len(files)))
            im = plt.imread(i)
            im = resize(im, resize_image)
            im = np.expand_dims(im, axis=0)
            preds = model.predict(im)
            im = np.squeeze(im)
            mask = np.round(np.squeeze(preds))
            if np.all(mask == 0):
                continue
            mask *= 255
            im *= 255
            imwrite(os.path.join(out_dir_image, os.path.basename(i)), im)
            imwrite(os.path.join(out_dir_mask, os.path.basename(i)), mask)



if __name__ == '__main__':

    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(544, 920, 3))
    base_model_output = base_model.get_layer('block3_conv3').output
    u1 = UpSampling2D()(base_model_output)
    concat_1 = Concatenate()([u1, base_model.get_layer('block2_conv2').output])
    conv_1 = Conv2D(128, 3, padding='same', activation='relu', name='first_trainable')(concat_1)
    conv_2 = Conv2D(128, 3, padding='same', activation='relu')(conv_1)
    u2 = UpSampling2D()(conv_2) # size 1080x1920
    concat_2 = Concatenate()([u2, base_model.get_layer('block1_conv2').output])
    conv_2 = Conv2D(64, 3, padding='same', activation='relu')(concat_2)
    conv_out = Conv2D(1, 3, padding='same', activation='sigmoid', name='sigmoid')(conv_2)

    model = Model(inputs=base_model.input, outputs=conv_out)
    trainable = False
    for layer in model.layers:
        if layer.name == 'up_sampling2d':
            trainable = True
        layer.trainable = trainable

    # model.summary()
    # size 270, 480, 256
    resize_image = (920, 544)
    resize_mask = (920, 544)
    train_path = 'train_data/'
    test_path = 'test_data'
    train_generator = DataGenerator(train_path, 2, resize=resize_image, resize_mask=resize_mask)
    test_generator = DataGenerator(test_path, 2, resize=resize_image, resize_mask=resize_mask)
    tb = TensorBoard()
    model.compile(Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    model_path = 'models/more_data_chkpt.h5'
    if not os.path.isfile(model_path):
        chpt = ModelCheckpoint(model_path, save_best_only=True, verbose=True)
        model.fit_generator(train_generator,
                epochs=1000,
                callbacks=[chpt],
                validation_data=test_generator,
                verbose=1
                )
        model.save('models/full.h5')
    else:
        model = tf.keras.models.load_model(model_path)
    test_generator = DataGenerator(test_path, 1, resize=resize_image, resize_mask=resize_mask)
    #conf_mat, _, _ = precision_and_recall_from_generator(test_generator, model)

    for i, (image, mask) in enumerate(test_generator):
        preds = np.squeeze(model.predict(image))
        fig, ax = plt.subplots()
        ax.imshow(image[0])
        ax.imshow(preds, alpha=0.3)
        plt.show()
