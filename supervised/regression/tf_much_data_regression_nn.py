import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

x_data = np.linspace(0, 10, 10 ** 6)

# Change up the noise here
noise = np.random.randn(len(x_data)) * 0.5

# y = mx + b
y_label = (0.5 * x_data) + 5 + noise

x_df = pd.DataFrame(data=x_data, columns=['X Data'])
y_df = pd.DataFrame(data=y_label, columns=['Y Label'])

dataset = pd.concat([x_df, y_df], axis=1)

print(dataset.head())
dataset.sample(500).plot(kind='scatter', x='X Data', y='Y Label')
plt.title('Items (Sample: 500)')
plt.show()

batch_size = 8

# Slope
m = tf.Variable(0.5)

# Bias
b = tf.Variable(0.8)

# Actual Values
x_placeholder = tf.placeholder(tf.float32, [batch_size])
y_placeholder = tf.placeholder(tf.float32, [batch_size])

# Prediction
y_model = m * x_placeholder + b

# Loss function
error = tf.reduce_sum(tf.square(y_placeholder - y_model))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    batches = 1000
    for i in range(batches):
        rand_ind = np.random.randint(len(x_data), size=batch_size)

        feed = {x_placeholder: x_data[rand_ind], y_placeholder: y_label[rand_ind]}

        sess.run(train, feed_dict=feed)

    model_m, model_b = sess.run([m, b])

print(model_m, model_b)

y_predicted_plot = model_m * x_data + model_b

dataset.sample(500).plot(kind='scatter', x='X Data', y='Y Label')
plt.plot(x_data, y_predicted_plot, 'r')
plt.title('Predicted plot (Sample: 500)')
plt.show()
