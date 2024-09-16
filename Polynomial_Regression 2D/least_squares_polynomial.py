from load_data import *

global type, degree
type = 'polynomial'

if type == 'polynomial':
    degree = 3

phi = []

def basis_func(index, x):
    if type == 'polynomial':
        return x ** index
    
for i in range(len(training_input)):
    row = []
    for j in range(degree + 1):
        row.append(basis_func(j, training_input[i][0]))
    phi.append(row)

phi = tf.convert_to_tensor(phi)

t1 = tf.linalg.matmul(phi, phi, transpose_a=True)
t2 = tf.linalg.matmul(phi, training_output, transpose_a=True)

weights = tf.linalg.solve(t1, t2)

error = training_output - tf.linalg.matmul(phi, weights)

variance = tf.linalg.matmul(error, error, transpose_a=True) / len(training_input)

print(weights, variance)