import tensorflow as tf

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('ResNet-L152.meta')
    saver.restore(sess, "ResNet-L152.ckpt")