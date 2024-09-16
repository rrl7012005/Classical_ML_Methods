from load_data import *


t1 = tf.linalg.matmul(training_input, training_input, transpose_a = True, transpose_b = False)
t2 = tf.linalg.matmul(training_input, training_output, transpose_a = True, transpose_b = False)

weights = tf.linalg.solve(t1, t2)

errors = training_output - tf.linalg.matmul(training_input, weights)

variance = tf.linalg.matmul(errors, errors, transpose_a = True) / n_samples

print(weights, variance)