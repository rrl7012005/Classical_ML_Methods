from least_squares_polynomial import *
import matplotlib.pyplot as plt
import numpy as np

def evaluate_points(points):

    list_of_tensors = []

    for i in range(degree + 1):
        list_of_tensors.append(points ** i)

    power_tensor = tf.concat(list_of_tensors, 1)

    f = tf.linalg.matmul(weights, power_tensor, transpose_a=True, transpose_b=True)

    return f

prediction = evaluate_points(testing_input)

rmse_error = tf.sqrt(tf.reduce_sum((prediction - testing_output) ** 2) / len(testing_output))

for i in range(len(prediction[0])):
    print("--------------------\n")
    print("The {}th prediction is: {} whereas the actual value is {} \n".format(i, prediction[0][i], testing_output[i][0]))
    print("\n")
    print("The true value was {} standard deviations away".format(abs(testing_output[i] - prediction[0][i])/int(tf.sqrt(variance[0][0]))))
    print("-------------------\n")

print("RMSE :", rmse_error)

graph = []
x = np.linspace(min(input) * 1.25, max(input) * 1.25 , 400)
y = evaluate_points(x)
sd_up = y + 2 * int(tf.sqrt(variance[0][0]))
sd_down = y - 2 * int(tf.sqrt(variance[0][0]))

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
plt.plot(tf.squeeze(x), tf.squeeze(y), color = 'b')
plt.plot(tf.squeeze(x), tf.squeeze(sd_up), color = 'r')
plt.plot(tf.squeeze(x), tf.squeeze(sd_down), color = 'r')
plt.legend()
plt.show()
