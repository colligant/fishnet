import numpy as np
import pdb
import os
from matplotlib.pyplot import imread, imshow, show
from tensorflow.keras.utils import Sequence
from glob import glob
from cv2 import resize

def _load_image(f, grayscale=False):
    im = imread(f)
    return im/np.max(im)


class ClassificationDataGenerator(Sequence):

    def __init__(self, data_directory, batch_size, balance_classes=False, resize=(),
            image_suffix='*.jpg'):
        self.classes = [d for d in os.listdir(data_directory) if \
                os.path.isdir(os.path.join(data_directory, d))]
        if len(self.classes) == 0:
            raise ValueError('no directories in data directory {}'.format(data_directory))
        self.batch_size = batch_size
        self.resize = resize
        self.image_suffix = image_suffix
        self.n_classes = len(self.classes)
        self.images = []
        self.labels = []
        self.n_instances = 0
        self.index_to_class = {}
        for idx, d in enumerate(self.classes):
            self.index_to_class[idx] = d
            image_files = glob(os.path.join(data_directory, d, self.image_suffix))
            self.images.extend(image_files)
            self.labels.extend([idx]*len(image_files))
            self.n_instances += len(image_files)
        self._shuffle()

    def _to_categorical(self, labels):
        one_hot = np.zeros((self.batch_size, self.n_classes))
        for i, ci in zip(labels, one_hot):
            ci[i] = 1
        return one_hot


    def __getitem__(self, idx):
        labels = self.labels[self.batch_size*idx:self.batch_size*(idx+1)]
        labels = self._to_categorical(labels)
        images = self.images[self.batch_size*idx:self.batch_size*(idx+1)]
        images = [_load_image(f) for f in images]
        if len(self.resize):
            for i in range(len(images)):
                images[i] = resize(images[i], self.resize) 
                # labels[i] = np.round(resize(labels[i], (0,0),
                #     fx=self.resize_mask[0], fy=self.resize_mask[1]))

        return np.asarray(images), np.asarray(labels)

    def _shuffle(self):
        indices = np.random.choice(np.arange(self.n_instances),
                self.n_instances, replace=False)
        self.labels = list(np.asarray(self.labels)[indices])
        self.images = list(np.asarray(self.images)[indices])

    def on_epoch_end(self):
        # shuffle data
        self._shuffle()

    def __len__(self):
        return int(np.ceil(self.n_instances // self.batch_size))


class SegmentationDataGenerator(Sequence):

    def __init__(self, data_directory, batch_size, resize=(), resize_mask=()):
        # resize should be tuple of (x_resize_factor, y_resize_factor)
        self.masks = sorted(glob(join(data_directory, 'masks', '*png')))
        self.images = sorted(glob(join(data_directory, 'images', '*png')))
        if len(self.masks) != len(self.images):
            raise ValueError('expected number of labels to equal number of images')
        self.n_instances = len(self.images)
        self.batch_size = batch_size
        self.resize = resize
        self.resize_mask = resize_mask


    def __getitem__(self, idx):
        labels = self.masks[self.batch_size*idx:self.batch_size*(idx+1)]
        images = self.images[self.batch_size*idx:self.batch_size*(idx+1)]
        images = [_load_image(f) for f in images]
        labels = [np.round(_load_image(f)) for f in labels]
        if len(self.resize):
            for i in range(len(images)):
                images[i] = resize(images[i], self.resize) 
                # images[i] = resize(images[i], (0,0), fx=self.resize[0],
                #         fy=self.resize[1]) 
                # resize uses some sort of interpolation, meaning
                # values of inputs get changed. For labels,
                # the values need to be 0 or 1.
                labels[i] = np.round(resize(labels[i], self.resize_mask))
                # labels[i] = np.round(resize(labels[i], (0,0),
                #     fx=self.resize_mask[0], fy=self.resize_mask[1]))

        return np.asarray(images), np.expand_dims(np.asarray(labels), -1)

    def on_epoch_end(self):
        # shuffle data
        indices = np.random.choice(np.arange(self.n_instances),
                self.n_instances, replace=False)
        self.masks = list(np.asarray(self.masks)[indices])
        self.images = list(np.asarray(self.images)[indices])

    def __len__(self):
        return int(np.ceil(self.n_instances // self.batch_size))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dg = ClassificationDataGenerator('../classification-images/', 1, resize=(548, 548))
    print(dg.index_to_class)
    for i, m in dg:
        plt.figure()
        plt.title(dg.index_to_class[np.argmax(m)])
        plt.imshow(np.squeeze(i))
        plt.show()

