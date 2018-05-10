#-*- conding:utf-8 -*-
import numpy as np
import tensorflow as tf
#1创建数据
np.random.seed(0)
n=100
x = np.linspace(0,6,n)+np.random.normal(loc=0.0, scale=2.0, size=n)#产生正态分布，均值为0方差为2
y = 14*x-7+np.random.normal(loc=0.0, scale=5.0, size=n)
#转化为矩阵
x.shape = -1,1
y.shape = -1,1

#2构建模型
#定义变量w和b
w = tf.Variable(initial_value=tf.random_uniform(shape=[1],minval=-1.0,maxval=1.0),name='w')
b = tf.Variable(initial_value=tf.zeros([1]),name="b")
print(w.shape)
#构建预测值y_hat
y_hat = w*x+b
print(y_hat)
#构建损失函数，以MSE为损失函数
loss = tf.reduce_mean(tf.square(y-y_hat),name='loss')
#以随机梯度下降优化损失函数，让损失函数最小
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train = optimizer.minimize(loss,name='train')
#更新全局变量
init_op=tf.global_variables_initializer()
#运行
def print_info(r_w,w_b,r_loss):
    print("w:{},b:{},loss:{}".format(r_w,w_b,r_loss))
with tf.Session() as sess:
    sess.run(init_op)
    r_w,w_b,r_loss = sess.run([w,b,loss])
    print_info(r_w,w_b,r_loss)
    #训练
    for i in range(50):
        sess.run(train)
        r_w, w_b, r_loss = sess.run([w, b, loss])
        print_info(r_w, w_b, r_loss)
