import numpy as np
from copy import deepcopy


def householder_transformation(a: np.array):
    v = deepcopy(a)
    
    v[0] = a[0] + np.sign(a[0].real) * np.linalg.norm(a)
    v = v / np.linalg.norm(v)
    H = np.eye(len(a), dtype=complex) - 2 * np.outer(v, v)

    return H

def QR_decomposition(A: np.array):
    n = A.shape[0]
    Q = np.eye(n, dtype=complex)
    R = A.astype(complex)
    
    for i in range(n - 1):
        H = np.eye(n, dtype=complex)
        H[i:, i:] = householder_transformation(R[i:, i])
        R = H @ R
        Q = Q @ H.T
    
    return Q, R

def solve_complex(a11, a12, a21, a22, epsilon = 1e-6):
    a = 1
    b = -a11 - a22
    c = a11 * a22 - a12 * a21
    d = b ** 2 - 4 * a * c

    if d > epsilon:
        return None, None
    
    d_c = complex(0, np.sqrt(-d))
    x1 = (-b + d_c) / (2 * a)
    x2 = (-b - d_c) / (2 * a)

    return x1, x2

def QR_eigenvalues(A: np.array, epsilon: float = 1e-6):
    n = A.shape[0]
    Ak = np.array(A, dtype=complex)
    eigenvalues = np.zeros(n, dtype=complex)
    iterations = 0
    
    while True:
        iterations += 1
        Q, R = QR_decomposition(Ak)
        Ak = R @ Q
        
        conv = True
        i = 0
        while (i < n):
            if i < n - 1 and np.abs(Ak[i + 1, i]) > epsilon:
                eigenvalue_1, eigenvalue_2 = solve_complex(Ak[i, i], Ak[i, i + 1], Ak[i + 1, i], Ak[i + 1, i + 1], epsilon)
                if eigenvalue_1 is not None and eigenvalue_2 is not None:
                    eigenvalues[i] = eigenvalue_1
                    eigenvalues[i + 1] = eigenvalue_2
                    i += 1
                else:
                    conv = False
            else:
                eigenvalues[i] = Ak[i, i]
            i += 1
        
        if conv:
            break
    
    return eigenvalues, iterations
            
    
def main():
    A = np.array([
        [-5, -8, 4],
        [4, 2, 6],
        [-2, 5, -6]
    ])
    print("Матрица A:")
    print(A)
    
    eigenvalues, iterations = QR_eigenvalues(A, 1e-9)
    
    print(f"Собственные значения:\n {eigenvalues}")
    print(f"Количество итераций: {iterations}")
    
    
    


if __name__ == "__main__":
    main()
