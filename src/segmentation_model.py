import tensorflow as tf
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import matplotlib.pyplot as plt
import numpy as np

from sys import stdout
from cv2 import resize, imwrite
from glob import glob
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, 
        UpSampling2D, Concatenate, Add)
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.metrics import confusion_matrix
from random import shuffle
import tensorflow.keras.layers as kl
import tensorflow.keras.models as km

from data_generators import SegmentationDataGenerator

def unet(input_size):
    inp = kl.Input(input_size)
    conv1 = kl.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inp)
    conv1 = kl.BatchNormalization()(conv1)
    conv1 = kl.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = kl.BatchNormalization()(conv1)
    pool1 = kl.MaxPooling2D(pool_size=(2, 2))(conv1)

    
    conv5 = kl.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv5 = kl.BatchNormalization()(conv5)
    conv5 = kl.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = kl.BatchNormalization()(conv5)


    depool9 = kl.UpSampling2D(size = (2,2))(conv5)
    upconv9 = kl.Conv2D(64, 2, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(depool9)
    merge9 = kl.concatenate([conv1,upconv9], axis = 3)
    conv9 = kl.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = kl.BatchNormalization()(conv9)
    conv9 = kl.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = kl.BatchNormalization()(conv9)
    conv10 = kl.Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs=inp, outputs=conv10)

    return model

def lr_schedule(epoch):
    lr = 1e-4
    if epoch > 80:
        lr /=16.
    elif epoch > 40:
        lr /= 8.
    elif epoch > 25:
        lr /= 4.
    elif epoch > 10:
        lr /= 2.
    print('Learning rate: ', lr)
    return lr

if __name__ == '__main__':

    #base_model = VGG16(include_top=False, weights='imagenet', input_shape=(None, None, 3))
    #base_model_output = base_model.get_layer('block3_conv3').output
    #u1 = UpSampling2D()(base_model_output)
    #concat_1 = Concatenate()([u1, base_model.get_layer('block2_conv2').output])
    #conv_1 = Conv2D(128, 3, padding='same', activation='relu', name='first_trainable')(concat_1)
    #conv_2 = Conv2D(128, 3, padding='same', activation='relu')(conv_1)
    #u2 = UpSampling2D()(conv_2) # size 1080x1920
    #concat_2 = Concatenate()([u2, base_model.get_layer('block1_conv2').output])
    #conv_2 = Conv2D(64, 3, padding='same', activation='relu')(concat_2)
    #conv_out = Conv2D(1, 3, padding='same', activation='sigmoid', name='sigmoid')(conv_2)

    #model = Model(inputs=base_model.input, outputs=conv_out)
    #trainable = False

    # model.summary()
    # size 270, 480, 256
    resize_image = (920, 544)
    resize_mask = (920, 544)
    model = unet((None, None, 3))
    train_path = '../segmentation-images/train_data/'
    test_path = '../segmentation-images/test_data/'
    data_gen_args = dict(rotation_range=0,
                         width_shift_range=0.0,
                         height_shift_range=0.0,
                         horizontal_flip=False,
                         zoom_range=0.0)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args,
            preprocessing_function=lambda x: x/255)

    # Provide the same seed and keyword arguments to the fit and flow methods

    bs = 2
    seed = 1
    # image_generator = image_datagen.flow_from_directory(
    #     '../segmentation-images/train_data/images/',
    #     target_size=resize_image,
    #     class_mode=None,
    #     batch_size=bs,
    #     seed=seed)

    # mask_generator = mask_datagen.flow_from_directory(
    #     '../segmentation-images/train_data/masks/',
    #     target_size=resize_image,
    #     class_mode=None,
    #     color_mode='grayscale',
    #     batch_size=bs,
    #     seed=seed)

    # # combine generators into one which yields image and masks
    # train_generator = (pair for pair in zip(image_generator, mask_generator))
    train_generator = SegmentationDataGenerator(train_path, 2, resize=resize_image, 
            resize_mask=resize_mask)
    test_generator = SegmentationDataGenerator(test_path, 2, resize=resize_image, 
            resize_mask=resize_mask)
    tb = TensorBoard()
    model.compile(Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    # model = tf.keras.models.load_model('pretrained.h5')
    model_out_path = 'models/seg-models/lowest_loss_checkpoint.h5'
    lr = LearningRateScheduler(lr_schedule)
    if not os.path.isfile(model_out_path):
        chpt = ModelCheckpoint(model_out_path, save_best_only=True, verbose=True, monitor='val_acc')
        model.fit_generator(train_generator,
                epochs=50,
                callbacks=[chpt, lr],
                validation_data=test_generator,
                verbose=1
                )
        model.save('models/seg-models/more_data_full.h5')
    else:
        model = tf.keras.models.load_model(model_out_path)
    test_generator = SegmentationDataGenerator(test_path, 1, resize=resize_image, resize_mask=resize_mask)
    for i, (image, mask) in enumerate(test_generator):
        preds = np.squeeze(model.predict(image))
        preds[preds < 0.7] = 0
        fig, ax = plt.subplots()
        ax.imshow(image[0])
        ax.imshow(preds, alpha=0.3)
        plt.show()
