from monte_carlo import mcarlo_expectation
import random
import math

#Goal: Now approximate P(Z >= 6.0). Answer: 2e-9

def distribution():
    return random.gauss(6, 1)

#Sampling function
def q(x):
    return math.exp(-0.5 * (x - 6) ** 2)/math.sqrt(math.pi * 2)

def p(x):
    y = math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)
    return y

#indicator function
def I(x):
    if x >= 6:
        return 1
    else:
        return 0
    
def f(x):
    return I(x) * p(x) / q(x)


print(mcarlo_expectation(f, 1000000, distribution))