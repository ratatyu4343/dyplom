import os
import numpy as np
import matplotlib.pyplot as plt
import random
from pysr import PySRRegressor


class KovasznayFlowSymbolRegressionModel:
    def __init__(self, model_name, X_file, y_file, niterations, binary_operators, unary_operators, npoints,
                 populations, population_size, extra_sympy_mappings={}, elementwise_loss="(prediction - target)^2", re=20):
        self.model_name = model_name
        self.ALL_X = np.load(X_file)
        self.ALL_Y = np.load(y_file)
        self.X = self.ALL_X[:npoints]
        self.y = self.ALL_Y[:npoints]
        self.niterations = niterations
        self.populations = populations
        self.population_size = population_size
        self.binary_operators = binary_operators
        self.unary_operators = unary_operators
        self.extra_sympy_mappings = extra_sympy_mappings
        self.elementwise_loss = elementwise_loss
        self.results_dir = f"./symbol_regression_statistic/{model_name}"
        self.Re = re
        self.nu = 1 / re
        self.l = 1 / (2 * self.nu) - np.sqrt(1 / (4 * self.nu ** 2) + 4 * np.pi ** 2)

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        self.model = PySRRegressor(
            model_selection="accuracy",
            populations=self.populations,
            population_size=self.population_size,
            niterations=self.niterations,
            binary_operators=self.binary_operators,
            unary_operators=self.unary_operators,
            equation_file="equations.txt",
            temp_equation_file=True,
            extra_sympy_mappings=self.extra_sympy_mappings,
            nested_constraints={"sin":{"sin":0, "cos":0},"cos":{"cos":0, "sin":0}, "exp":{"exp":0}, "sqrt":{"sqrt":0}}
        )

    def u_function(self, x):
        return 1 - np.exp(self.l * x[:, 0:1]) * np.cos(2 * np.pi * x[:, 1:2])

    def v_function(self, x):
        return self.l / (2 * np.pi) * np.exp(self.l * x[:, 0:1]) * np.sin(2 * np.pi * x[:, 1:2])

    def p_function(self, x):
        return 1 / 2 * (1 - np.exp(2 * self.l * x[:, 0:1]))

    def train(self):
        self.model.fit(self.X, self.y)
        print(self.model)

    def save_results(self):
        # Save model
        model_file = os.path.join(self.results_dir, "model.txt")
        with open(model_file, "w") as f:
            f.write(str(self.model))

        # Save predictions
        predictions = self.model.predict(self.ALL_X)
        np.save(os.path.join(self.results_dir, "predictions.npy"), predictions)

        u_pred = predictions[:, 0]
        v_pred = predictions[:, 1]
        p_pred = predictions[:, 2]

        u_exact = self.u_function(self.ALL_X).reshape(-1)
        v_exact = self.v_function(self.ALL_X).reshape(-1)
        p_exact = self.p_function(self.ALL_X).reshape(-1)

        velocity_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)
        velocity_exact = np.sqrt(u_exact ** 2 + v_exact ** 2)

        velocity_difference = np.abs(velocity_pred - velocity_exact)
        pressure_difference = np.abs(p_pred - p_exact)

        vel_vmin = min(min(velocity_pred), min(velocity_exact))
        vel_vmax = max(max(velocity_pred), max(velocity_exact))

        press_vmin = min(min(p_pred), min(p_exact))
        press_vmax = max(max(p_pred), max(p_exact))

        # Save plots
        if not os.path.exists(f'{self.results_dir}/plots'):
            os.makedirs(f'{self.results_dir}/plots')

        # Виведення скалярної швидкості
        plt.figure(figsize=(10, 5))
        plt.scatter(self.ALL_X[:, 0], self.ALL_X[:, 1], c=velocity_pred, vmin=vel_vmin, vmax=vel_vmax, cmap='viridis')
        plt.colorbar(label='Передбачена магнітуда швидкості')
        plt.title('Передбачена швидкість')
        plt.savefig(f'{self.results_dir}/plots/vel_predicted.png')
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.scatter(self.ALL_X[:, 0], self.ALL_X[:, 1], c=velocity_exact, vmin=vel_vmin, vmax=vel_vmax, cmap='viridis')
        plt.colorbar(label='Розрахована магнітуда швидкості')
        plt.title('Розрахована швидкість')
        plt.savefig(f'{self.results_dir}/plots/vel_numeric.png')
        plt.close()

        # Обчислення тиску для передбачених та аналітичних значень
        plt.figure(figsize=(10, 5))
        plt.scatter(self.ALL_X[:, 0], self.ALL_X[:, 1], c=p_pred, vmin=press_vmin, vmax=press_vmax, cmap='viridis')
        plt.colorbar(label='Передбачена магнітуда тиску')
        plt.title('Передбачений тиск')
        plt.savefig(f'{self.results_dir}/plots/press_predicted.png')
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.scatter(self.ALL_X[:, 0], self.ALL_X[:, 1], c=p_exact, vmin=press_vmin, vmax=press_vmax, cmap='viridis')
        plt.colorbar(label='Розрахована магнітуда тиску')
        plt.title('Розрахований тиск')
        plt.savefig(f'{self.results_dir}/plots/press_numeric.png')
        plt.close()

        # Виведення модуля різниці
        plt.figure(figsize=(10, 5))
        plt.scatter(self.ALL_X[:, 0], self.ALL_X[:, 1], c=velocity_difference, cmap='coolwarm')
        plt.colorbar(label='Різниця швидкості (абсолютна)')
        plt.title('Різниця швидкості (абсолютна)')
        plt.savefig(f'{self.results_dir}/plots/eror_vel.png')
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.scatter(self.ALL_X[:, 0], self.ALL_X[:, 1], c=pressure_difference, cmap='coolwarm')
        plt.colorbar(label='Різниця тиску (абсолютна)')
        plt.title('Різниця тиску (абсолютна)')
        plt.savefig(f'{self.results_dir}/plots/eror_press.png')
        plt.close()
