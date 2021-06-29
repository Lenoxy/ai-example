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

feature_columns = [tf.feature_column.numeric_column('x', shape=[1])]

estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_label, test_size=0.3, random_state=101)

# 3/7 split
print(x_train.shape)
print(x_test.shape)

input_function = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train, batch_size=8, num_epochs=None, shuffle=True)

train_input_function = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train, batch_size=8, num_epochs=1000, shuffle=False)

test_input_function = tf.estimator.inputs.numpy_input_fn({'x': x_test}, y_test, batch_size=8, num_epochs=1000, shuffle=False)

estimator.train(input_fn=input_function, steps=1000)


train_metrics = estimator.evaluate(input_fn=train_input_function, steps=1000)
print(train_metrics)

test_metrics = estimator.evaluate(input_fn=test_input_function, steps=1000)
print(test_metrics)

# Use the trained model in production
x_production = np.linspace(0, 10, 10)
print(x_production)

input_function_predict = tf.estimator.inputs.numpy_input_fn({'x': x_production}, shuffle=False)

y_production_iterate = list(estimator.predict(input_fn=input_function_predict))
y_production = []
for dict in y_production_iterate:
    y_production.append(dict.get('predictions'))


print(y_production)

plt.plot(x_train, y_train, 'g,')
plt.plot(x_test, y_test, 'y,')
plt.plot(x_production, y_production, 'b*')
plt.title('Plot (g: Train data, y: Test data, b: Simulated production data)')

plt.show()


