import tensorflow as tf
from data_preparing import read_img,getbatch

test_path = 'D:\\DL_data\\test\\'
w = 78
h = 78
c = 1

print("reading test set")
x_test,y_test = read_img(test_path,w,h,c)
with tf.Session() as sess:
    # load graph
    saver = tf.train.import_meta_graph('.\\checkpoint_dir\\MyModel.meta')
    saver.restore(sess, tf.train.latest_checkpoint('.\\checkpoint_dir'))
    graph = tf.get_default_graph()

    x = graph.get_operation_by_name('x').outputs[0]
    y_ = graph.get_operation_by_name('y_').outputs[0]
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    acc = graph.get_tensor_by_name('acc:0')
    # test
    test_acc, n_batch = 0, 0

    for x_test_a, y_test_a in getbatch(x_test, y_test, 1500, shuffle=True):
        ac = sess.run(acc, feed_dict={x: x_test_a,
                                      y_: y_test_a,
                                      keep_prob: 1.0})
        test_acc += ac
        n_batch += 1
    print("   test acc: %f" % (test_acc / n_batch))
