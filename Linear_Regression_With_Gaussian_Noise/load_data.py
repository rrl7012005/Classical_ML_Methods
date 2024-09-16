import tensorflow as tf
import pandas as pd

data = pd.read_csv('real_estate.csv')

print(data.head())

tensor_data = tf.constant(data)
tensor_data = tf.cast(tensor_data, dtype=tf.float32)
tensor_data = tf.random.shuffle(tensor_data)

output = tensor_data[:,-1]
input_1 = tensor_data[:,2:5]
ones = tf.ones([len(input_1),1], tf.dtypes.float32)

input = tf.concat([ones,input_1], 1)

output = tf.expand_dims(output, -1)

n_samples = len(input)

r_train = 0.95
r_test = 0.05

training_input = input[:int(r_train * n_samples),:]
training_output = output[:int(r_train * n_samples),:]

testing_input = input[int(r_train * n_samples):, :]
testing_output = output[int(r_train * n_samples):,:]

print(training_input.shape, testing_input.shape, n_samples)