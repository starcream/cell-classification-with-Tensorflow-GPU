from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np

# training set : 8701 ,validation set : 2175 , six kinds of cells
train_path = 'D:\\DL_data\\training\\'
val_path   = 'D:\\DL_data\\validation\\'
# each image : 78 * 78 * 1
w = 78
h = 78
c = 1
n_epoch = 130
batch_size = 400

# function to read image ,return imgs and labels
def read_img(path):
    category = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    for idx, folder in enumerate(category):
        print(idx,'   ',folder)
        for im in glob.glob(folder + '\\*.png'):
            #print('reading the images:%s' % (im))
            img = io.imread(im)
            img = transform.resize(img, (w, h, c))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)

# read data
print("read training set")
train_data, train_label = read_img(train_path)
print("reading validation set")
val_data , val_label    = read_img(val_path)
# random shuffle training set
num_example = train_data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
x_train = train_data[arr]
y_train = train_label[arr]
# random shuffle validation set
num_example = val_data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
x_val = val_data[arr]
y_val = val_label[arr]

# -----------------build network----------------------
x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

# conv 1& pool1（78——>39)
conv1 = tf.layers.conv2d(
    inputs=x,
    filters=32,
    kernel_size=[4, 4],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# conv2 & pool2 (39->13)
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[4, 4],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=3)

re1 = tf.reshape(pool2, [-1, 13 * 13 * 64])

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
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# get batch of data
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
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


# train and validate

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for epoch in range(n_epoch):
    print("epoch : ",epoch)
    # training
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err;
        train_acc += ac;
        n_batch += 1
    #print("   train loss: %f" % (train_loss / n_batch))
    print("   train acc: %f" % (train_acc / n_batch))

    # validation
    if(epoch%20 == 9):
        val_loss, val_acc, n_batch = 0, 0, 0
        for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
            err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a})
            val_loss += err;
            val_acc += ac;
            n_batch += 1
        #print("   validation loss: %f" % (val_loss / n_batch))
        print("   validation acc: %f" % (val_acc / n_batch))

sess.close()