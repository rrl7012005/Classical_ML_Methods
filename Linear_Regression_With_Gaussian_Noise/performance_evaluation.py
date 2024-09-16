from least_squares import *
import math


model_output = tf.linalg.matmul(weights, testing_input, transpose_a = True, transpose_b = True)
model_output = tf.squeeze(model_output, 0)
testing_output = tf.squeeze(testing_output, -1)
testing_output = tf.transpose(testing_output)


for i in range(len(model_output)):
    print("--------------------\n")
    print("The {}th prediction is: {} whereas the actual value is {} \n".format(i, model_output[i], testing_output[i]))
    print("\n")
    print("The true value was {} standard deviations away".format(abs(testing_output[i] - model_output[i])/math.sqrt(variance[0][0])))
    print("-------------------\n")

rms_testing_loss = tf.sqrt(tf.reduce_sum((model_output - testing_output) ** 2))

print("Root mean Square error", rms_testing_loss)