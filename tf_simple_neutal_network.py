import numpy as np
import tensorflow.compat.v1 as tf
from matplotlib import pyplot as plt

tf.disable_v2_behavior()

x_data = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)

y_label = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)

plt.plot(x_data, y_label, '*')
plt.title('Linear points with noise')
plt.show()

# y = mx + b

# Usually random IRL, since the neural network will adjust these params to "fix itself"
m = tf.Variable(0.40)
b = tf.Variable(0.79)

error = 0

for x, y in zip(x_data, y_label):
    # y_hat -> predicted value
    y_hat = m * x + b

    # Check difference between predicted and actual to fix later
    # Square to punish higher errors (Syntax: **2)
    error += (y - y_hat) ** 2

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
# Tell the optimizer, the smaller the variable error, the better
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    training_steps = 100
    for i in range(training_steps):
        sess.run(train)

    final_slope, final_intercept = sess.run([m, b])

x_test = np.linspace(-1, 11, 10)

y_predicted_plot = final_slope * x_test + final_intercept

plt.plot(x_data, y_label, '*')
plt.plot(x_test, y_predicted_plot, 'r')
plt.title('Predicted plot')
plt.show()
