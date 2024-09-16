from bayesian import *
import matplotlib.pyplot as plt
import numpy as np
import math


def evaluate_points(points):

    list_of_tensors = []

    for i in range(degree + 1):
        list_of_tensors.append(points ** i)

    power_tensor = tf.concat(list_of_tensors, 1)

    pred_mean = tf.linalg.matmul(power_tensor, mean_vector, transpose_a=False, transpose_b=False)

    t1 = tf.linalg.matmul(power_tensor, covariance_mat, transpose_a=False)
    pred_var = tf.linalg.matmul(t1, power_tensor, transpose_b = True)

    pred_var += noise_var * tf.eye(pred_var.shape[0])
    pred_var = tf.linalg.diag_part(pred_var)

    return pred_mean, pred_var

predicted_mean, predicted_var = evaluate_points(testing_input)

rmse_error = tf.sqrt(tf.reduce_sum((predicted_mean - testing_output) ** 2) / len(testing_output))

for i in range(len(predicted_mean)):
    print("--------------------\n")
    print("The {}th prediction is: {} whereas the actual value is {} \n".format(i, predicted_mean[i], testing_output[i][0]))
    print("\n")
    print("The true value was {} standard deviations away".format(abs(testing_output[i] - predicted_mean[i])/tf.sqrt(predicted_var[i])))
    print("-------------------\n")

print("RMSE :", rmse_error)

#Compute marginal likelihood

def compute_marg(training_output, phi):
    term1 = - math.log(2 * math.pi) * len(training_output) / 2

    cov_1 = tf.linalg.matmul(phi, phi, transpose_b=True) / prior_lambda
    cov = cov_1 + (noise_var + 1e-6) * tf.eye(cov_1.shape[0])

    eigenvalues = tf.abs(tf.linalg.eigvalsh(cov))

    term2 = - tf.reduce_sum(tf.math.log(eigenvalues)) / 2
    cov_inv = tf.linalg.inv(cov)

    term3_1 = tf.linalg.matmul(training_output, cov_inv, transpose_a=True)
    term3 = tf.linalg.matmul(term3_1, training_output)

    return term1 + term2 + term3


print("THE MARGINAL LIKELIHOOD IS: ", compute_marg(training_output, phi))

graph = []
x = np.linspace(min(input), max(input), 400)
mean, var = evaluate_points(x)
# print(mean, var)
sd_up = mean + 2 * tf.sqrt(var)
sd_down = mean - 2 * tf.sqrt(var)

plt.figure()
plt.title('Initial Scatter Plot')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.scatter(tf.squeeze(training_input), tf.squeeze(training_output), color = 'green', marker = 'x')
plt.show()

plt.figure()
plt.title("Regression Plot")
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.scatter(tf.squeeze(training_input), tf.squeeze(training_output), color = 'green', marker = 'x')
plt.plot(tf.squeeze(x), tf.squeeze(mean), color = 'b')
plt.plot(tf.squeeze(x), tf.squeeze(sd_up), color = 'r')
plt.plot(tf.squeeze(x), tf.squeeze(sd_down), color = 'r')
plt.legend()
plt.show()
