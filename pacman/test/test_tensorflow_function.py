import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

tf.set_random_seed(2)
a = tf.random_normal(shape=[3,3,6,16], stddev=0.01)

with tf.Session() as sess:
    print(sess.run(a))
