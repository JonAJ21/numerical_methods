import numpy as np
from copy import deepcopy


def jacobi(
    A: np.array,
    epsilon: float = 1e-6
):
    iterations = 0
    n = A.shape[0]
    current_A = deepcopy(A)
    total_U = np.eye(n)
    max_element_change = 1e18
    
    while max_element_change > epsilon:
        max_element_change = 0
        max_i, max_j = 0, 0
        
        for i in range(n):
            for j in range(n):
                if abs(current_A[i][j]) > max_element_change and i != j:
                    max_element_change = abs(current_A[i][j])
                    max_i, max_j = i, j
                    
        if max_element_change > epsilon:
            phi = np.arctan((2 * current_A[max_i][max_j]) / (current_A[max_i][max_i] - current_A[max_j][max_j])) / 2

            U = np.eye(n)
            
            U[max_i][max_i] = np.cos(phi)
            U[max_j][max_j] = np.cos(phi)
            U[max_i][max_j] = -np.sin(phi)
            U[max_j][max_i] = np.sin(phi)
            
            current_A = U.T @ current_A @ U
            total_U = total_U @ U
        
        iterations += 1
        
    eigenvalues = np.diagonal(current_A)
    eigenvectors = [total_U[:, i] for i in range(n)]
    
    return eigenvalues, eigenvectors, iterations
    
    

def main():
    A = np.array([
        [4, 7, -1],
        [7, -9, -6],
        [-1, -6, -4]
    ])
    print("Матрица A:")
    print(A)
    
    eigenvalues, eigenvectors, iterations = jacobi(A)
    
    print(f"Собственные значения:\n {eigenvalues}")
    print("Собственные векторы:")
    for v in eigenvectors: 
        print(v)
    print(f"Количество итераций: {iterations}")
    
    
    print("Проверка:")
    print("Ax = λx")
    
    print(A @ eigenvectors[0], " = ",  eigenvalues[0] * eigenvectors[0])
    print(A @ eigenvectors[1], " = ", eigenvalues[1] * eigenvectors[1])
    print(A @ eigenvectors[2], " = ", eigenvalues[2] * eigenvectors[2])
    
    
    


if __name__ == "__main__":
    main()
