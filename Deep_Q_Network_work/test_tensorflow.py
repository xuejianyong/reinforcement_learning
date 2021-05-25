import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

"""
v1 = tf.placeholder(tf.float32, shape=[2, 3, 4])
#print(v1.name)
#print(type(v1))
v1 = tf.placeholder(tf.float32, shape=[2, 3, 4], name='ph')
#print(v1.name)
#print(type(v1))
v1 = tf.placeholder(tf.float32, shape=[2, 3, 4], name='ph')
#print(v1.name)
#print(type(v1))
#print('-------------------------------------------------------')
v2 = tf.Variable([1, 2], dtype=tf.float32)
#print(type(v2))
#print(v2.name)
v2 = tf.Variable([1,2], dtype=tf.float32,name='V')
#print(type(v2))
#print(v2.name)
v2 = tf.Variable([1,2], dtype=tf.float32,name='V')
#print(type(v2))
#print(v2.name)
#print('-------------------------------------------------------')
v3 = tf.get_variable(name='gv', shape=[])
#print(type(v3))
#print(v3.name)
#v4 = tf.get_variable(name='gv', shape=[2])
#print(type(v4))
#print(v4.name)
#print('-------------------------------------------------------')
#vs = tf.trainable_variables()
#print(len(vs))
#for v in vs:
#    print(v)
#    print(type(v))
#    print(v.name)
#    print()

# name scope and the variable scope
with tf.name_scope('nsc1'):
    v1 = tf.Variable([1],name='v1')
    with tf.variable_scope('vsc1'):
        v2 = tf.Variable([1], name='v2')
        v3 = tf.get_variable(name='v3', shape=[])

print('-------------------------------------------')
print(type(v1))
print(v1.name)
print(type(v2))
print(v2.name)
print('-------------------------------------------')
print(type(v3))
print(v3.name)
print('-------------------------------------------')
with tf.name_scope('nsc1'):
    v4 = tf.Variable([1], name='v4')
    v5 = tf.get_variable(name='v5', shape=[])
print(type(v4))
print(v4.name)
print('-------------------------------------------')
print(type(v5))
print(v5.name)


def my_image_filter():
    conv1_weights = tf.Variable(tf.random.normal([5,5,32,32]),name='conv1_weights')
    conv1_biases = tf.Variable(tf.zeros([32]), name='conv1_biases')
    conv2_weights = tf.Variable(tf.random.normal([5,5,32,32]), name='conv2_weights')
    conv2_biases = tf.Variable(tf.zeros([32]), name='conv2_biases')
    return None

result1 = my_image_filter()
result2 = my_image_filter()
vs = tf.trainable_variables()
for v in vs:
    print(v)


# another example
def conv_relu(kernel_shape, bias_shape):
    weights = tf.get_variable('weights', kernel_shape,initializer=tf.random_normal_initializer())
    biases = tf.get_variable('biases', bias_shape, initializer=tf.constant_initializer(0.0))
    return None

def my_image_filter():
    with tf.variable_scope("conv1"):
        relu1 = conv_relu([5,5,32,32], [32])
    with tf.variable_scope("conv2"):
        return conv_relu([5,5,32,32], [32])


with tf.variable_scope("image_filter") as scope:
    result1 = my_image_filter()
    scope.reuse_variables()
    result2 = my_image_filter()

vs = tf.trainable_variables()
for v in vs:
    print(v)
    print(v.name)


with tf.variable_scope('eval_net'):
    with tf.variable_scope('l1'):
        c_name = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
        w1 = tf.get_variable(name='w1', shape=[])
        b1 = tf.get_variable(name='b1', shape=[])

    with tf.variable_scope('l2'):
        c_name = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
        w1 = tf.get_variable(name='w2', shape=[])
        b1 = tf.get_variable(name='b2', shape=[])

with tf.variable_scope('target_net'):
    with tf.variable_scope('l1'):
        c_name = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
        w1 = tf.get_variable(name='w1', shape=[])
        b1 = tf.get_variable(name='b1', shape=[])

    with tf.variable_scope('l2'):
        c_name = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
        w1 = tf.get_variable(name='w2', shape=[])
        b1 = tf.get_variable(name='b2', shape=[])
"""
a = [10, 10]
b = [20, 20]
c = [30, 30]
print(np.hstack((a,b,c)))
































