import numpy as np

def gaussian_elimination(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)

    for i in range(n):
        max_row = i + np.argmax(np.abs(A[i:, i]))
        if A[max_row, i] == 0:
            raise ValueError("Matrix is singular.")

        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]

        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    x = np.zeros_like(b)
    for i in range(n - 1, -1, -1):
        if A[i, i] == 0:
            raise ValueError("Zero on diagonal during back substitution.")
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]

    return x

def F(x: np.array, a: int) -> np.array:
    return np.array([
        x[0]**2 + x[1]**2 - a**2, 
        x[0] - np.exp(x[1]) + a  
    ])
    
def Jacobian_F(x: np.array) -> np.array:
    return np.array([
        [2*x[0], 2*x[1]],
        [1, -np.exp(x[1])]
    ])

def Phi(x: np.array, a: int, tao: float = 0.1) -> np.array:
    return tao * F(x, a) + x

def Jacobian_Phi(x: np.array) -> np.array:
    return np.array([
        [2*x[0] + 1, 2 * x[1]],
        [1, -np.exp(x[1]) + 1]
    ])
    
def sufficient_condition_for_iteration_method_convergence(x, tao: float = 0.1) -> bool:
    J = Jacobian_Phi(x)
    if all(sum(tao * np.abs(J[i])) < 1 for i in range(len(J))):
        return True
    return False
    
def simple_iterations(x0, a, eps=1e-4, max_iter=1000, tao=0.1):
    x = np.array(x0, dtype=float)
    
    iteration = 0
    while iteration < max_iter:
        iteration += 1
        new_x = Phi(x, a, tao)
        if not sufficient_condition_for_iteration_method_convergence(x, tao):
            print("The sufficient condition for convergence is not satisfied.")
            return iteration, None
        if np.linalg.norm(new_x - x) < eps:
            return iteration, new_x
        x = new_x
    return iteration, x

def newton_method(x0, a, eps=1e-4, max_iter=1000):
    x = np.array(x0, dtype=float)
    
    iteration = 0
    while iteration < max_iter:
        iteration += 1
        J = Jacobian_F(x)
        F_val = F(x, a)
        delta_x = gaussian_elimination(J.copy(), -F_val.copy())#np.linalg.solve(J, -F_val)
        new_x = x + delta_x
        if np.linalg.norm(abs(new_x - x)) < eps:
            return iteration, x
        x = new_x
    return iteration, x

def main():
    a = 1
    
    x0_simple = np.array([-0.5, -0.5])
    i, x = simple_iterations(x0_simple, a, tao=0.1)
    print(f"Solution: {x}, Iterations: {i}")
    
    x0_newton = np.array([-0.5, -0.5])
    i, x = newton_method(x0_newton, a)
    print(f"Solution: {x}, Iterations: {i}")


if __name__ == "__main__":
    main()
        
    