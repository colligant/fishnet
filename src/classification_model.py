import tensorflow as tf
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.layers as kl

from sys import stdout
from cv2 import resize, imwrite
from glob import glob
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.metrics import confusion_matrix
from random import shuffle

from data_generators import ClassificationDataGenerator

# flatten = kl.Flatten()(last_conv)
# fc_1 = kl.Dense(32, name='fc_1', activation='relu')(flatten)
# fc_out = kl.Dense(n_classes, name='last_fc', activation='softmax')(fc_1)

if __name__ == '__main__':

    n_classes = 4
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model_output = base_model.get_layer('block4_conv3')
    last_conv = kl.Conv2D(64, 3, name='conv_last', activation='relu')(base_model_output.output)
    gap = kl.GlobalAveragePooling2D()(last_conv)
    fc_out = kl.Dense(n_classes, activation='softmax', name='softmax')(gap)
    model = Model(inputs=base_model.input, outputs=fc_out)
    trainable = False
    for layer in model.layers:
        if layer.name == 'conv_last':
            trainable = True
        layer.trainable = trainable

    # model.summary()
    model_dir = 'models/class-models/'
    resize_image = (224, 224)
    train_path = '../classification-images/train/'
    test_path = '../classification-images/test/'
    train_generator = ClassificationDataGenerator(train_path, 8, resize=resize_image)
    test_generator = ClassificationDataGenerator(test_path, 8, resize=resize_image)
    tb = TensorBoard()
    model.compile(Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model_path = model_dir + 'gap_cam_shallower_network_cropped_data.h5'
    if not os.path.isfile(model_path):
        chpt = ModelCheckpoint(model_path, save_best_only=True, verbose=True, monitor='val_acc')
        model.fit_generator(train_generator,
                epochs=5,
                callbacks=[chpt],
                validation_data=test_generator,
                verbose=1
                )
        model.save(model_dir + 'full.h5')
    else:
        model = tf.keras.models.load_model(model_path)
    softmax = model.get_layer('softmax')
    last_conv = model.get_layer('conv_last')
    cam_model = Model(inputs=model.input, outputs=(last_conv.output, softmax.output))
    test_generator = ClassificationDataGenerator(test_path, 1, resize=resize_image)
    idx_to_class = test_generator.index_to_class
    y_p = []
    y_t = []
    for i, (image, m) in enumerate(test_generator):
        last_conv, probs = cam_model.predict(image)
        y_pred = np.argmax(probs)
        y_true = np.argmax(m)
        y_p.append(y_pred)
        y_t.append(y_true)
        output_weights = softmax.get_weights()[0][:, y_pred]
        filters = last_conv[0]
        cam = np.zeros((filters.shape[0], filters.shape[1]))
        for i, weight in enumerate(output_weights):
            cam += filters[:, :, i]*abs(weight)
        cam = resize(cam, (224, 224))
        fig, ax = plt.subplots(ncols=2)
        # image = image.astype(np.uint8)
        ax[0].imshow(image[0])
        ax[1].imshow(cam)
        plt.title('predicted to be {}, actually {}'.format(idx_to_class[np.argmax(m)],
            idx_to_class[y_pred]))
        plt.show()
# {0: 'adult_cutthroat', 1: 'unknown', 2: 'juvenile_grayling', 3: 'juvenile_cutthroat'}
# acc 0.98
# [[75  0  0  0]
#  [ 0 17  2  0]
#  [ 0  0 21  0]
#  [ 0  1  0 59]]
print(idx_to_class)
y_p = np.array(y_p)
y_t = np.array(y_t)
print('acc', np.sum(y_p == y_t)/(len(y_t)))
conf_mat = confusion_matrix(y_t, y_p)
print(conf_mat)


