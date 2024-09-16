from load_data import *


#Gaussian prior with covariance matrix 
global mean_vector, covariance_mat, noise_var, prior_lambda

prior_mean = 0
prior_lambda = 0.01 #lambda = 1/var_prior

noise_var = 100

degree = 10

phi = []

def basis_func(index, x):
    return x ** index
    
for i in range(len(training_input)):
    row = []
    for j in range(degree + 1):
        row.append(basis_func(j, training_input[i][0]))
    phi.append(row)

phi = tf.convert_to_tensor(phi)

t1 = tf.linalg.matmul(phi, phi, transpose_a=True, transpose_b=False) / noise_var
t2 = t1 + tf.eye(t1.shape[0]) * prior_lambda
covariance_mat = tf.linalg.inv(t2)

t3 = tf.linalg.matmul(phi, training_output, transpose_a=True) / noise_var
mean_vector = tf.linalg.matmul(covariance_mat, t3)

print(covariance_mat, mean_vector)