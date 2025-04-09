import numpy as np

def F(x):
    """Система уравнений"""
    return np.array([
        x[0]**2 + x[1]**2 - 16,  # x² + y² - 16 = 0
        x[0] - np.exp(x[1]) + 4    # x - eʸ + 4 = 0
    ])

def J(x):
    """Якобиан системы"""
    return np.array([
        [2*x[0], 2*x[1]],     # ∂F₁/∂x, ∂F₁/∂y
        [1, -np.exp(x[1])]     # ∂F₂/∂x, ∂F₂/∂y
    ])

def simple_iteration(x0, epsilon, max_iter=1000):
    counter = 0
    x1, x2 = x0
    
    while counter < max_iter:  # Защита от бесконечного цикла
        counter += 1
        
        try:
            # Вычисляем новые значения
            x1_new = np.sqrt(4**2 - x2**2)
            x2_new = np.log(x1_new + 4)
            
            # Проверка сходимости
            if np.sqrt((x1_new - x1)**2 + (x2_new - x2)**2) < epsilon:
                return np.array([x1_new, x2_new]), counter
            
            x1, x2 = x1_new, x2_new
            
        except (ValueError, RuntimeWarning):
            # При ошибке возвращаем текущие значения
            return np.array([x1, x2]), counter
    
    return np.array([x1, x2]), counter

def newton_method(x0, epsilon, max_iter=100):
    """Метод Ньютона"""
    x = x0.copy()
    for i in range(max_iter):
        try:
            delta_x = np.linalg.solve(J(x), -F(x))
        except np.linalg.LinAlgError:
            break
        x += delta_x
        if np.linalg.norm(delta_x) < epsilon:
            return x, i+1
    return x, max_iter
def main():
    initial_guesses = [
        np.array([4.0, 4.0]),    # Для положительного корня
    ]

    print("Решения системы уравнений:")
    for i, x0 in enumerate(initial_guesses):
        print(f"\nНачальное приближение {i+1}: {x0}")
        
        # Метод простой итерации
        sol_si, iter_si = simple_iteration(x0, 1e-6)
        print("Метод простой итерации:")
        print(f"x = {sol_si[0]:.6f}, y = {sol_si[1]:.6f}")
        print(f"Итераций: {iter_si}")

        # Метод Ньютона
        sol_n, iter_n = newton_method(x0, 1e-6)
        print("\nМетод Ньютона:")
        print(f"x = {sol_n[0]:.6f}, y = {sol_n[1]:.6f}")
        print(f"Итераций: {iter_n}")
    
if __name__ == "__main__":
    main()