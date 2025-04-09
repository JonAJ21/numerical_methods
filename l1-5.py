import numpy as np

# ---------------------------
# Хаусхолдер + Хессенберг
# ---------------------------

def householder_reflection(x):
    """Создание вектора Хаусхолдера"""
    v = x.astype(complex)
    normx = np.linalg.norm(v)
    if normx == 0:
        return np.zeros_like(v)
    sign = 1 if v[0] == 0 else v[0] / abs(v[0])
    v[0] += sign * normx
    v /= np.linalg.norm(v)
    return v

def hessenberg_manual(A):
    """Преобразование A к верхней Хессенберговой форме"""
    A = A.astype(complex)
    n = A.shape[0]
    Q_total = np.eye(n, dtype=complex)

    for k in range(n - 2):
        x = A[k+1:, k]
        v = householder_reflection(x)
        H = np.eye(n, dtype=complex)

        H_k = np.eye(n - k - 1, dtype=complex) - 2.0 * np.outer(v, v.conj())
        H[k+1:, k+1:] = H_k

        A = H @ A @ H.conj().T
        Q_total = Q_total @ H

    return A, Q_total

# ---------------------------
# QR с Вилкинсоном
# ---------------------------

def wilkinson_shift(H):
    """Сдвиг Вилкинсона по нижнему 2x2 блоку"""
    d = (H[-2, -2] - H[-1, -1]) / 2
    sign = np.sign(d) if d != 0 else 1
    mu = H[-1, -1] - (sign * H[-1, -2]**2) / (abs(d) + np.sqrt(d**2 + H[-1, -2]**2))
    return mu

def qr_algorithm_complex(A, tol=1e-12, max_iter=1000, verbose=False):
    """QR-алгоритм со сдвигами, поддержка комплексных λ"""
    H, _ = hessenberg_manual(A)
    n = H.shape[0]
    iter_count = 0

    while iter_count < max_iter:
        # Проверка сходимости
        if np.allclose(np.tril(H, -1), 0, atol=tol):
            break

        mu = wilkinson_shift(H)
        Q, R = np.linalg.qr(H - mu * np.eye(n))
        H = R @ Q + mu * np.eye(n)
        iter_count += 1

        if verbose and iter_count % 10 == 0:
            print(f"Итерация {iter_count}, ||нижняя часть|| = {np.linalg.norm(np.tril(H, -1)):.2e}")

    # Извлечение λ: учёт комплексных значений через 2×2 блоки
    eigenvalues = []
    i = 0
    while i < n:
        if i < n - 1 and abs(H[i + 1, i]) > tol:
            a, b = H[i, i], H[i, i + 1]
            c, d = H[i + 1, i], H[i + 1, i + 1]
            trace = a + d
            det = a * d - b * c
            disc = trace**2 - 4 * det
            sqrt_disc = np.sqrt(disc)
            λ1 = (trace + sqrt_disc) / 2
            λ2 = (trace - sqrt_disc) / 2
            eigenvalues.extend([λ1, λ2])
            i += 2
        else:
            eigenvalues.append(H[i, i])
            i += 1

    return np.array(eigenvalues), iter_count

# ---------------------------
# Пример использования
# ---------------------------

if __name__ == "__main__":
    A = np.array([
        [-5, -8, 4],
        [4, 2, 6],
        [-2, 5, -6]
    ], dtype=float)

    eigenvalues, iterations = qr_algorithm_complex(A, tol=1e-12, verbose=True)

    print(f"\nСобственные значения (итераций: {iterations}):")
    for val in eigenvalues:
        print(val)
        
    print(np.linalg.eigvals(A))