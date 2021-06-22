import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

# x = tf.constant(10)
#
# y = tf.constant(20)
#
# print(type(x))
# print(type(y))
#
# with tf.Session() as sess:
#     result = sess.run(x + y)
#
# print(result)



# const = tf.constant(10)
#
# fill_mat = tf.fill((4, 4), 10)
#
# zeros = tf.zeros((4, 4))
#
# ones = tf.ones((4, 4))
#
# randn = tf.random_normal((4, 4))
#
# randu = tf.random_uniform((4, 4), minval=0, maxval=1)
#
# ops = [
#     const,
#     fill_mat,
#     zeros,
#     ones,
#     randn,
#     randu
# ]
#
# with tf.Session() as sess:
#     for op in ops:
#         print(sess.run(op))


# n1 = tf.constant(1)
# n2 = tf.constant(2)
#
# n3 = n1 + n2
#
# with tf.Session() as sess:
#     result = sess.run(n3)
#
# print(result)
# print(tf.get_default_graph)

with tf.Session() as sess:
    tensor = tf.random_uniform((4,4), minval=0, maxval=1)
    print(tensor)
    var = tf.Variable(initial_value=tensor)
    print(var)
    print('\n')
    init = tf.global_variables_initializer()

    print(sess.run(init))
    print(sess.run(var))

    placeholder = tf.placeholder(tf.float32)

    



