from typing import Callable
import numpy as np


def F1(x: np.array, a: int) -> np.array:
    if x[1] >= 0:
        return np.array([
            x[0] - np.exp(x[1]) + a,
            x[1] - np.sqrt(a ** 2 - x[0] ** 2)
        ])
    else:
        return np.array([
            x[0] - np.exp(x[1]) + a,
            x[1] + np.sqrt(a ** 2 - x[0] ** 2)
        ])
    
def Phi1(x: np.array, a: int) -> np.array:
    if x[1] >= 0:
        return np.array([
            np.exp(x[1]) - a,
            np.sqrt(a ** 2 - x[0] ** 2)
        ])
    else:
        return np.array([
            np.exp(x[1]) - a,
            -np.sqrt(a ** 2 - x[0] ** 2)
        ])

def Jacobian_Phi1(x: np.array, a: int) -> np.array:
    if x[1] >= 0:
        return np.array([
            [0, np.exp(x[1])],
            [-x[0] / np.sqrt(a ** 2 - x[0] ** 2), 0]
        ])
    else:
        return np.array([
            [0, np.exp(x[1])],
            [x[0] / np.sqrt(a ** 2 - x[0] ** 2), 0]
        ])

def F2(x: np.array, a: int) -> np.array:
    if x[0] >= 0:
        return np.array([
            x[0] - np.sqrt(a ** 2 - x[1] ** 2),
            x[1] - np.log(x[0] + 1)
        ])
    else:
        return np.array([
            x[0] + np.sqrt(a ** 2 - x[1] ** 2),
            x[1] - np.log(x[0] + 1)
        ])

def Phi2(x: np.array, a: int) -> np.array:
    if x[0] >= 0:
        return np.array([
            np.sqrt(a ** 2 - x[1] ** 2),
            np.log(x[0] + 1)
        ])
    else:
        return np.array([
            -np.sqrt(a ** 2 - x[1] ** 2),
            np.log(x[0] + 1)
        ])
        
def Jacobian_Phi2(x: np.array, a: float) -> np.array:
    if x[0] >= 0:
        return np.array([
            [0, -x[1] / np.sqrt(a ** 2 - x[1] ** 2)],
            [1 /(x[0] + a), 0]
        ])
    else:
        return np.array([
            [0, x[1] / np.sqrt(a ** 2 - x[1] ** 2)],
            [1 /(x[0] + a), 0]
        ])


def sufficient_condition_for_iteration_method_convergence(jacobian: Callable, x: np.array, a: float, tao: float = 0.1) -> bool:
    J = jacobian(x, a)
    return all(sum(tao * np.abs(J[i])) < 1 for i in range(len(J)))
    
    
def simple_iterations(phi: Callable, jacobian: Callable, x0: np.array, a: float, eps=1e-3, max_iter=1000, tao=1):
    x = np.array(x0, dtype=float)
    
    iteration = 0
    while iteration < max_iter:
        iteration += 1
        new_x = tao * phi(x, a)
        #print(iteration, (x, F(x, a)), (new_x, F(x, a)))
        if not sufficient_condition_for_iteration_method_convergence(jacobian, x, a, tao):
            print("The sufficient condition for convergence is not satisfied.")
            return iteration, x
        if np.linalg.norm(new_x - x) < eps:
            return iteration, new_x
        x = new_x
    return iteration, x

def main():
    a = 4
    
    x0_simple = np.array([-0.5, -0.5])
    i, x = simple_iterations(Phi1, Jacobian_Phi1, x0_simple, a, tao=0.2, eps=1e-9)
    print("Simple iterations:")
    print(f"Solution: {x}, Iterations: {i}")
    
    print("Tests:")
    print(f"Simple iterations: {F1(x, a)} == [0, 0]\n\n")
    
    x0_simple = np.array([0.5, 0.5])
    i, x = simple_iterations(Phi2, Jacobian_Phi2, x0_simple, a, eps=1e-9)
    print("Simple iterations:")
    print(f"Solution: {x}, Iterations: {i}")
    
    print("Tests:")
    print(f"Simple iterations: {F2(x, a)} == [0, 0]")
    print() 



if __name__ == "__main__":
    main()
        
    