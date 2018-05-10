#-*- conding:utf-8 -*-
'''
1.图的构建
2.session的用法
3.变量的定义以及初始化，必须把把初始化的运行操作放到回话里面
4.feed为操作赋值（类似于定义函数之后再传入值），fetch获取数据流中的参数变量
'''
import tensorflow as tf
a = tf.constant([[1,2],[3,4]],dtype=tf.int32)
print(type(a))
b = tf.constant([5,6,7,8],shape=[2,2])#shape指构建的大小
p = tf.matmul(a,b)#矩阵乘法
r = tf.add(a,b)
print(type(p))
print(a.graph is tf.get_default_graph())#判断a是否在默认图中
#新建图
# graph = tf.Graph()
# with graph.as_default():
#     #在这个代码中用的就是新的图中
#     d=tf.constant(5.4)
#     print(d.graph is tf.get_default_graph())
#     print(d.graph is graph)

#session
#启动
# sess = tf.Session()
# # print(sess)
# result = sess.run(fetches=[p,r])#如果传递的fetches是一个列表，那么返回值是一个list集合,fetches：表示获取那个op操作的结果值
# print("type:{},value:{}".format(type(result),result))
# #关闭
# sess.close()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess2:
    print("value:{}".format(sess2.run(fetches=[p,r])))
v = tf.Variable(3.0,dtype=tf.float32)
co = tf.constant(2,dtype=tf.float32)
re = tf.add(v,co)
#注意变量进行计算的时候最好带好初始化w1.initialized_value() * a
w1 = tf.Variable(tf.random_normal(shape=[10], stddev=0.5, seed=28, dtype=tf.float32), name='w1')
k = tf.constant(value=2.0, dtype=tf.float32)
w2 = tf.Variable(w1.initialized_value() * k, name='w2')
inint_op = tf.global_variables_initializer()
print(inint_op)
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(inint_op)
    print("value{}".format(sess.run(w2)))
#给定占位符placeholder
# 构建一个矩阵的乘法，但是矩阵在运行的时候给定
m1 = tf.placeholder(dtype=tf.float32, shape=[2, 3], name='placeholder_1')
m2 = tf.placeholder(dtype=tf.float32, shape=[3, 2], name='placeholder_2')
m3 = tf.matmul(m1, m2)

with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    print("result:\n{}".format(
        sess.run(fetches=m3, feed_dict={m1: [[1, 2, 3], [4, 5, 6]], m2: [[9, 8], [7, 6], [5, 4]]})))
    print("result:\n{}".format(m3.eval(feed_dict={m1: [[1, 2, 3], [4, 5, 6]], m2: [[9, 8], [7, 6], [5, 4]]})))
