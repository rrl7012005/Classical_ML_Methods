import numpy as np

arrays = np.array([[0, 0.2, 0.5], [0.3, 0.1, 0.2], [0.8, 0.4, 0.3]])
print(arrays)
arrays /= np.array([1, 2, 3])

print(arrays)