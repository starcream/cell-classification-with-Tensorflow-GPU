from skimage import io, transform
import glob
import os
import tensorflow as tf
import numpy as np

# training set : 8701 ,validation set : 2175 , six kinds of cells
train_path = 'D:\\DL_data\\training\\'
val_path = 'D:\\DL_data\\validation\\'
# each image : 78 * 78 * 1
w = 78
h = 78
c = 1
n_epoch = 130
batch_size = 400


# function to read image ,shuffle,return imgs and labels
def read_img(path):
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


# read data
print("read training set")
x_train, y_train = read_img(train_path)
print("reading validation set")
x_val, y_val = read_img(val_path)

# -----------------build network----------------------
x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

# conv 1& pool1（78*78*3 -> 78*78*32)
conv1 = tf.layers.conv2d(
    inputs=x,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
# 78*78*32 -》 39*39*32
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# conv2 & pool2 (39*39*32->39*39*64)
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=3)

# conv3 & pool3 (13->5)
conv3 = tf.layers.conv2d(
    inputs=pool2,
    filters=128,
    kernel_size=[4, 4],
    padding="valid",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
)
pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

# dropout layer
keep_prob = tf.placeholder(tf.float32)
drop = tf.nn.dropout(pool3, keep_prob)

re1 = tf.reshape(drop, [-1, 5 * 5 * 128])

# fcn , l2 reg
dense1 = tf.layers.dense(inputs=re1,
                         units=1024,
                         activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
dense2 = tf.layers.dense(inputs=dense1,
                         units=512,
                         activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
logits = tf.layers.dense(inputs=dense2,
                         units=6,
                         activation=None,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
# ---------------------------end of network---------------------------

loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)
train_op = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9, beta2=0.999).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# train and validate

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for epoch in range(n_epoch):
    print("epoch : ", epoch)
    # training
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in getbatch(x_train, y_train, batch_size, shuffle=True):
        _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a,
                                                                y_: y_train_a,
                                                                keep_prob: 0.5})
        train_loss += err
        train_acc += ac
        n_batch += 1
    # print("   train loss: %f" % (train_loss / n_batch))
    print("   train acc: %f" % (train_acc / n_batch))

    # validation
    if (epoch % 20 == 19):
        val_loss, val_acc, n_batch = 0, 0, 0
        for x_val_a, y_val_a in getbatch(x_val, y_val, batch_size, shuffle=False):
            err, ac = sess.run([loss, acc], feed_dict={x: x_val_a,
                                                       y_: y_val_a,
                                                       keep_prob: 1.0})
            val_loss += err
            val_acc += ac
            n_batch += 1
        # print("   validation loss: %f" % (val_loss / n_batch))
        print("   validation acc: %f" % (val_acc / n_batch))

sess.close()

# how to save model and read from model
# how to adjust learning rate while training  -- using AdamOptimizer
# how to achieve data augmentation
