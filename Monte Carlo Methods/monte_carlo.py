#Goal: Approximate P(0 <= Z <= 1.96) for a 2D standard normal distribution (0.475)

import random
import math

def mcarlo_volume(function, domain, maximum, n_iter):

    if isinstance(domain[0], list):
        dimension = len(domain)
    else:
        dimension = 1
        domain = [domain]

    accepts = 0
    for n in range(n_iter):
        function_inputs = []
        domain_volume = 1

        for d in range(dimension):
            function_inputs.append(random.uniform(domain[d][0], domain[d][1]))
            domain_volume *= abs(domain[d][1] - domain[d][0])

        v = random.uniform(0, maximum)

        if v <= function(function_inputs):
            accepts += 1

    return accepts * maximum * domain_volume / n_iter

def mcarlo_expectation(function, n_iter, distribution):

    total = 0

    for n in range(n_iter):
        x = distribution()
        total += function(x)
    
    expectation = total / n_iter
    return expectation

def f(inp):
    x = inp[0]
    return math.exp(-0.5 * x ** 2)/math.sqrt(2 * math.pi)

domain = [0, 3]
maximum = 1/math.sqrt(2 * math.pi)
mcarlo_estimate = mcarlo_volume(f, domain, maximum, 1000000)

print("AREA APPROXIMATION IS {}".format(mcarlo_estimate))