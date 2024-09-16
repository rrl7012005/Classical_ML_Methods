from monte_carlo import mcarlo_expectation
from importance_sampling import *
import random
import math

def g(x):
    return x

def Cov(x, y):
    total = 0
    n = len(x)
    for i in range(n):
        for j in range(n):
            total += (x[i] - x[j]) * (y[i] - y[j])

    total /= 2 * (n ** 2)

    return total

N = 10000000

Eg_estimate = mcarlo_expectation(g, N, distribution)
Ef_estimate = mcarlo_expectation(f, N, distribution)

# n = 100

# Eg_estimates = [mcarlo_expectation(g, N, distribution) for _ in range(n)]
# Ef_estimates = [mcarlo_expectation(f, N, distribution) for _ in range(n)]

# c = - Cov(Ef_estimates, Eg_estimates) * N

c = -2.5099966528882176e-10

Improved_MC_estimate = Ef_estimate + c * (Eg_estimate - 6)

print(Improved_MC_estimate)