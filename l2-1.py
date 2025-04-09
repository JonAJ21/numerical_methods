import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return x**3 + x**2 - 2*x - 1

def df(x):
    return 3*x**2 + 2*x - 2

def simple_iteration(epsilon):
    x = 1.5
    lambda_ = 0.1
    max_iter = 1000
    for i in range(max_iter):
        x_new = x - lambda_ * f(x)
        if abs(x_new - x) < epsilon:
            return x_new, i+1
        x = x_new
    return x, max_iter

def newton_method(epsilon):
    x0 = 1.5
    f = x0**3 + x0**2 - 2*x0 - 1
    f_prime = 3*x0**2 + 2*x0 - 2
    x1 = x0 - f / f_prime
    iterations = 1
    while abs(x1 - x0) > epsilon:
        x0 = x1
        f = x0**3 + x0**2 - 2*x0 - 1
        f_prime = 3*x0**2 + 2*x0 - 2
        x1 = x0 - f / f_prime
        iterations += 1
    return x1, iterations


    
    
def main():
    eps = 1e-9
    root_iter, iter_count = simple_iteration(eps)
    root_newton, newton_count = newton_method(eps)

    print(f"Метод простой итерации: корень = {root_iter:.6f}, итераций = {iter_count}")
    print(f"Метод Ньютона: корень = {root_newton:.6f}, итераций = {newton_count}")


if __name__ == "__main__":
    main()