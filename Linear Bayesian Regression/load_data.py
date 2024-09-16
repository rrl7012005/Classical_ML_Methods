import tensorflow as tf
import pandas as pd

data = pd.read_csv('random3.csv')

print(data.head())

tensor_data = tf.constant(data)
tensor_data = tf.cast(tensor_data, dtype=tf.float32)
tensor_data = tf.random.shuffle(tensor_data, seed=12)

output = tensor_data[:,-1]
input = tensor_data[:,0]

output = tf.expand_dims(output, -1)
input = tf.expand_dims(input, -1)

n_samples = len(input)

r_train = 0.8
r_test = 0.2

training_input = input[:int(r_train * n_samples),:]
training_output = output[:int(r_train * n_samples),:]

testing_input = input[int(r_train * n_samples):, :]
testing_output = output[int(r_train * n_samples):,:]

print(training_input.shape, testing_input.shape, n_samples, training_input, training_output)