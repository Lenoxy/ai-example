import os

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

tf.disable_v2_behavior()

diabetes_data = pd.read_csv('./testdata/diabetes.csv')

cols_to_normalize = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
                     'Insulin', 'BMI', 'Pedigree']

# Normalization
diabetes_data[cols_to_normalize] = diabetes_data[cols_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

number_pregnant = tf.feature_column.numeric_column('Number_pregnant')
glucose_concentration = tf.feature_column.numeric_column('Glucose_concentration')
blood_pressure = tf.feature_column.numeric_column('Blood_pressure')
triceps = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
pedigree = tf.feature_column.numeric_column('Pedigree')
# If all the possible values are known, this is very helpful
group = tf.feature_column.categorical_column_with_vocabulary_list('Group', ['A', 'B', 'C', 'D'])
# Otherwise use Tensorflow to create these categories automatically (hash_bucket_size is the max different categories)
# group = tf.feature_column.categorical_column_with_hash_bucket('Group', hash_bucket_size=10)
age_bucket = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column('Age'),
    boundaries=[20, 30, 40, 50, 60, 70, 80]
)


embedded_group_column = tf.feature_column.embedding_column(group, dimension=4)

feature_columns = [
    number_pregnant, glucose_concentration, blood_pressure, triceps,
    insulin, bmi, pedigree, embedded_group_column, age_bucket
]

# Train test split
x_data = diabetes_data.drop('Class', axis=1)
y_label = diabetes_data['Class']

x_train, x_test, y_train, y_test = train_test_split(x_data, y_label, train_size=0.7)

# Train

dnn_input_function = tf.estimator.inputs.pandas_input_fn(
    x=x_test,
    y=y_test,
    batch_size=10,
    num_epochs=1000,
    shuffle=True
)
dnn_model = tf.estimator.DNNClassifier(hidden_units=[10, 10, 10], feature_columns=feature_columns, n_classes=2)

print('###### Train metrics  ######')
# It seems that increasing the steps to 10000 helps a lot, takes a lot longer too though
train_metrics = dnn_model.train(dnn_input_function, steps=1000)

# Test

dnn_test_input_function = tf.estimator.inputs.pandas_input_fn(
    x=x_test,
    y=y_test,
    batch_size=10,
    num_epochs=1000,
    shuffle=True
)

test_metrics = dnn_model.evaluate(dnn_test_input_function)
print('###### Test metrics ######')
print(test_metrics)

# Predict values


predict_input_function = tf.estimator.inputs.pandas_input_fn(
    x=x_test[0],
    batch_size=10,
    num_epochs=1,
    shuffle=False
)

prediction_results = dnn_model.predict(predict_input_function)

print(list(prediction_results))

