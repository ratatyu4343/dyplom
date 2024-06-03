import os

os.environ["DDE_BACKEND"] = "tensorflow"
import numpy as np
import deepxde as dde
from pysr import PySRRegressor

# Параметри проблеми
Re = 20
nu = 1 / Re
l = 1 / (2 * nu) - np.sqrt(1 / (4 * nu**2) + 4 * np.pi**2)

# Аналітичні розв'язки для швидкостей та тиску
def u_func(x):
    return 1 - np.exp(l * x[:, 0:1]) * np.cos(2 * np.pi * x[:, 1:2])

def v_func(x):
    return l / (2 * np.pi) * np.exp(l * x[:, 0:1]) * np.sin(2 * np.pi * x[:, 1:2])

def p_func(x):
    return 1 / 2 * (1 - np.exp(2 * l * x[:, 0:1]))

# Генерація даних для аналітичних розв'язків
num_points = 100000
spatial_domain = dde.geometry.Rectangle(xmin=[-0.5, -0.5], xmax=[1, 1.5])
X = spatial_domain.random_points(num_points)

# Обчислення аналітичних розв'язків для кожної точки
u_exact = u_func(X).reshape(-1)
v_exact = v_func(X).reshape(-1)
p_exact = p_func(X).reshape(-1)

# Підготовка даних для моделі PySRRegressor
y = np.vstack((u_exact, v_exact, p_exact)).T

# Створення та навчання моделі PySRRegressor
model = PySRRegressor(
    niterations=40,  # < Збільште для кращих результатів
    binary_operators=["+", "*"],
    unary_operators=["cos", "exp", "sin"],
    equation_file="equations.txt",  # Збереження рівнянь в файл
    temp_equation_file=True,
)

print(X.shape, y.shape)
model.fit(X, y)
print(model)

# Виведення збережених рівнянь
equations = model.equations_
for eq in equations:
    print(eq)