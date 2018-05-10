#-*- conding:utf-8 -*-
import tensorflow as tf
'''
变量作用域
'''
#1正常形式
def my_func(x):
    w1 = tf.Variable(tf.random_normal([1]))[0]
    b1 = tf.Variable(tf.random_normal([1]))[0]
    r1 = w1 * x + b1

    w2 = tf.Variable(tf.random_normal([1]))[0]
    b2 = tf.Variable(tf.random_normal([1]))[0]
    r2 = w2 * r1 + b2

    return r1, w1, b1, r2, w2, b2


# 下面两行代码还是属于图的构建
x = tf.constant(3, dtype=tf.float32)
r = my_func(x)

with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    # 初始化
    tf.global_variables_initializer().run()
    # 执行结果
    print(sess.run(r))
