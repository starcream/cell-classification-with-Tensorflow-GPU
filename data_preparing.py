from skimage import io, transform
import glob
import os
import numpy as np

# function to read image ,shuffle,return imgs and labels


def read_img(path, w, h, c):
    category = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    for idx, folder in enumerate(category):
        print(idx, '   ', folder)
        # get images and labels in certain dir
        for im in glob.glob(folder + '\\*.png'):
            img = io.imread(im)
            img = transform.resize(img, (w, h, c))
            imgs.append(img)
            labels.append(idx)
    data = np.asarray(imgs, np.float32)
    labels = np.asarray(labels, np.int32)
    # shuffle images and label
    num = data.shape[0]
    arr = np.arange(num)
    np.random.shuffle(arr)
    data = data[arr]
    labels = labels[arr]
    return data, labels


# get batch of data
def getbatch(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]