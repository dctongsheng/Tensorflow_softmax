#-*- conding:utf-8 -*-
#实现累加操作变量的更新用到tf.assign(ref=x, value=x + 1)
#1定义一个变量
import tensorflow as tf
# x = tf.Variable(0,dtype=tf.int32,name="v_x")
# #2变量的更新
# assign_op = tf.assign(ref=x,value=x+1)
# #3变量的初始化
# x_inint_op = tf.global_variables_initializer()
# #4运行
# with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
#     sess.run(x_inint_op)
#     for i in range(5):
#
#         r_x=sess.run(x)
#         print(r_x)
#         sess.run(assign_op)

#实现维度的更新
# x = tf.Variable(
#     initial_value=[],
#     dtype=tf.float32,
#     trainable=False,#是否加载到内存的缓存区中间，这种情况下需要单独进行管理，默认为ture
#     validate_shape=False#形状进行改变
# )
# concat = tf.concat([x,[0.0,0.0]],axis=0)
# assign_op=tf.assign(x,concat,validate_shape=False)
# x_inint_op=tf.global_variables_initializer()
# with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
#     sess.run(x_inint_op)
#     for i in range(5):
#         r_x=sess.run(x)
#         print(r_x)
#         sess.run(assign_op)

#3实现阶乘
# s = tf.Variable(1,dtype=tf.int32)
# i = tf.placeholder(dtype=tf.int32)
# su=s*i
# assign_op=tf.assign(s,su)
# x_inint_op=tf.global_variables_initializer()
# with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
#     sess.run(x_inint_op)
#     for j in range(1,6):
#         sess.run(assign_op,feed_dict={i:j})
#         r_x = sess.run(s)
#     print(r_x)

#正常的做法  通过control_dependencies可以指定依赖关系，这样的话，就不用管内部的更新操作了，更加方便
#控制依赖
sum = tf.Variable(1,dtype=tf.int32)
i = tf.placeholder(dtype=tf.int32)
tmp_sum=sum*i
assign_op=tf.assign(sum,tmp_sum)
with tf.control_dependencies([assign_op]):
    # 如果需要执行这个代码块中的内容，必须先执行control_dependencies中给定的操作/tensor
    sum = tf.Print(sum, data=[sum, sum.read_value()], message='sum:')
x_inint_op=tf.global_variables_initializer()
with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    sess.run(x_inint_op)
    for j in range(1,6):
        # sess.run(assign_op,feed_dict={i:j})
        r_x = sess.run(sum,feed_dict={i:j})
    print(r_x)


