import os
os.environ["DDE_BACKEND"] = "tensorflow"

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

import imageio
from PIL import Image

import re
import pandas as pd

import custom_callback

class KovasznayFlowModel:
    def __init__(self, model_name, num_domain_points, num_boundary_points, num_test_points, num_show_points,
                 network_structure, geometry, num_iterations, lr=1e-4, re=20, save_animation=False):
        self.model_name = f"./models_statistic/{model_name}"
        self.num_domain_points = num_domain_points
        self.num_boundary_points = num_boundary_points
        self.num_test_points = num_test_points
        self.num_show_points = num_show_points
        self.network_structure = network_structure
        self.geometry = geometry
        self.num_iterations = num_iterations
        self.lr = lr
        self.save_animation = save_animation

        self.Re = re
        self.nu = 1 / re
        self.l = 1 / (2 * self.nu) - np.sqrt(1 / (4 * self.nu**2) + 4 * np.pi**2)

        self.X = self.geometry.random_points(self.num_show_points)
        self.predict_imgs_for_animation = []
        self.save_prediction_images_callback = (
            custom_callback.SavePredictionsCallback(100, self.X, self.predict_imgs_for_animation)
        )

        # Create a directory for the model if it doesn't exist
        if not os.path.exists("models_statistic"):
            os.makedirs("models_statistic")
        if not os.path.exists(self.model_name):
            os.makedirs(self.model_name)

        self.build_model()

    def pde(self, x, u):
        u_vel, v_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:]
        u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
        u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)
        u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
        u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)

        v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)
        v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
        v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
        v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)

        p_x = dde.grad.jacobian(u, x, i=2, j=0)
        p_y = dde.grad.jacobian(u, x, i=2, j=1)

        momentum_x = (
                u_vel * u_vel_x + v_vel * u_vel_y + p_x - 1 / self.Re * (u_vel_xx + u_vel_yy)
        )
        momentum_y = (
                u_vel * v_vel_x + v_vel * v_vel_y + p_y - 1 / self.Re * (v_vel_xx + v_vel_yy)
        )
        continuity = u_vel_x + v_vel_y

        return [momentum_x, momentum_y, continuity]

    def u_function(self, x):
        return 1 - np.exp(self.l * x[:, 0:1]) * np.cos(2 * np.pi * x[:, 1:2])

    def v_function(self, x):
        return self.l / (2 * np.pi) * np.exp(self.l * x[:, 0:1]) * np.sin(2 * np.pi * x[:, 1:2])

    def p_function(self, x):
        return 1 / 2 * (1 - np.exp(2 * self.l * x[:, 0:1]))

    def build_model(self):
        # Create boundary conditions
        def boundary_outflow_condition(x, on_boundary):
            return on_boundary and dde.utils.isclose(x[0], 1)

        boundary_condition_u = dde.icbc.DirichletBC(
            self.geometry, self.u_function, lambda _, on_boundary: on_boundary, component=0
        )
        boundary_condition_v = dde.icbc.DirichletBC(
            self.geometry, self.v_function, lambda _, on_boundary: on_boundary, component=1
        )
        boundary_condition_right_p = dde.icbc.DirichletBC(
            self.geometry, self.p_function, boundary_outflow_condition, component=2
        )

        # Create PDE data object
        self.data = dde.data.PDE(
            self.geometry,
            self.pde,
            [boundary_condition_u, boundary_condition_v, boundary_condition_right_p],
            num_domain=self.num_domain_points,
            num_boundary=self.num_boundary_points,
            num_test=self.num_test_points,
        )

        # Create neural network
        self.net = dde.nn.FNN(self.network_structure, "tanh", "Glorot normal")
        self.model = dde.Model(self.data, self.net)

    def train_model(self):
        self.model.compile("adam", lr=0.1)
        losshistory, train_state = self.model.train(
            iterations=100,
            callbacks=[self.save_prediction_images_callback] if self.save_animation else [],
            display_every=100
        )

        self.model.compile("adam", lr=self.lr)
        losshistory, train_state = self.model.train(
            iterations=self.num_iterations,
            callbacks=[self.save_prediction_images_callback] if self.save_animation else [],
            display_every=10000
        )

        # Save training history
        if not os.path.exists(f'{self.model_name}/train_history'):
            os.makedirs(f'{self.model_name}/train_history')
        dde.saveplot(losshistory, train_state, issave=True, isplot=False, output_dir=f'{self.model_name}/train_history')

        # Save training metrics and plots
        self.save_metrics(train_state)

        # Save animation
        if self.save_animation:
            self.create_and_save_animation()

        # Save predictions
        self.save_predictions()

    def save_metrics(self, train_state):
        output = self.model.predict(self.X)

        u_pred = output[:, 0]
        v_pred = output[:, 1]
        p_pred = output[:, 2]

        u_exact = self.u_function(self.X).reshape(-1)
        v_exact = self.v_function(self.X).reshape(-1)
        p_exact = self.p_function(self.X).reshape(-1)

        f = self.model.predict(self.X, operator=self.model.data.pde)

        l2_difference_u = dde.metrics.l2_relative_error(u_exact, u_pred)
        l2_difference_v = dde.metrics.l2_relative_error(v_exact, v_pred)
        l2_difference_p = dde.metrics.l2_relative_error(p_exact, p_pred)

        residual = np.mean(np.absolute(f))

        metrics = {
            "Mean residual": residual,
            "L2 relative error in u": l2_difference_u,
            "L2 relative error in v": l2_difference_v,
            "L2 relative error in p": l2_difference_p
        }

        # Save metrics to a file
        with open(f"{self.model_name}/metrics.txt", "w") as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")

        # Plot and save results
        self.plot_results(self.X, u_pred, v_pred, p_pred, u_exact, v_exact, p_exact)

    def plot_results(self, X, u_pred, v_pred, p_pred, u_exact, v_exact, p_exact):
        if not os.path.exists(f'{self.model_name}/plots'):
            os.makedirs(f'{self.model_name}/plots')

        # Compute magnitude of velocity
        velocity_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)
        velocity_exact = np.sqrt(u_exact ** 2 + v_exact ** 2)
        velocity_difference = np.abs(velocity_pred - velocity_exact)
        pressure_difference = np.abs(p_pred - p_exact)

        vel_vmin = min(min(velocity_pred), min(velocity_exact))
        vel_vmax = max(max(velocity_pred), max(velocity_exact))

        press_vmin = min(min(p_pred), min(p_exact))
        press_vmax = max(max(p_pred), max(p_exact))

        # Виведення скалярної швидкості
        plt.figure(figsize=(10, 5))
        plt.scatter(X[:, 0], X[:, 1], c=velocity_pred, vmin=vel_vmin, vmax=vel_vmax, cmap='viridis')
        plt.colorbar(label='Передбачена магнітуда швидкості')
        plt.title('Передбачена швидкість')
        plt.savefig(f'{self.model_name}/plots/vel_predicted.png')
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.scatter(X[:, 0], X[:, 1], c=velocity_exact, vmin=vel_vmin, vmax=vel_vmax, cmap='viridis')
        plt.colorbar(label='Розрахована магнітуда швидкості')
        plt.title('Розрахована швидкість')
        plt.savefig(f'{self.model_name}/plots/vel_numeric.png')
        plt.close()

        # Обчислення тиску для передбачених та аналітичних значень
        plt.figure(figsize=(10, 5))
        plt.scatter(X[:, 0], X[:, 1], c=p_pred, vmin=press_vmin, vmax=press_vmax, cmap='viridis')
        plt.colorbar(label='Передбачена магнітуда тиску')
        plt.title('Передбачений тиск')
        plt.savefig(f'{self.model_name}/plots/press_predicted.png')
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.scatter(X[:, 0], X[:, 1], c=p_exact, vmin=press_vmin, vmax=press_vmax, cmap='viridis')
        plt.colorbar(label='Розрахована магнітуда тиску')
        plt.title('Розрахований тиск')
        plt.savefig(f'{self.model_name}/plots/press_numeric.png')
        plt.close()

        # Виведення модуля різниці
        plt.figure(figsize=(10, 5))
        plt.scatter(X[:, 0], X[:, 1], c=velocity_difference, cmap='coolwarm')
        plt.colorbar(label='Різниця швидкості (абсолютна)')
        plt.title('Різниця швидкості (абсолютна)')
        plt.savefig(f'{self.model_name}/plots/eror_vel.png')
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.scatter(X[:, 0], X[:, 1], c=pressure_difference, cmap='coolwarm')
        plt.colorbar(label='Різниця тиску (абсолютна)')
        plt.title('Різниця тиску (абсолютна)')
        plt.savefig(f'{self.model_name}/plots/eror_press.png')
        plt.close()

    def create_and_save_animation(self):
        if not os.path.exists(f"{self.model_name}/animation"):
            os.makedirs(f"{self.model_name}/animation")

        velocity_magnitudes_prediction = []
        for UV in self.predict_imgs_for_animation:
            u_pred = UV[:, 0]
            v_pred = UV[:, 1]
            velocity_magnitudes_prediction.append(np.sqrt(u_pred ** 2 + v_pred ** 2))

        vmin = min(map(lambda x: min(x), velocity_magnitudes_prediction))
        vmax = max(map(lambda x: max(x), velocity_magnitudes_prediction))

        for i, velocity_magnitudes in enumerate(velocity_magnitudes_prediction):
            try:
                plt.figure(figsize=(10, 5))
                plt.scatter(self.X[:, 0], self.X[:, 1], c=velocity_magnitudes, vmin=vmin, vmax=vmax, cmap='viridis')
                plt.colorbar(label='Передбачена магнітуда швидкості')
                plt.title('Передбачена швидкість')
                plt.savefig(f"{self.model_name}/animation/{i}.png")
                plt.close()
            except:
                print("some eror")

        def atoi(text):
            return int(text) if text.isdigit() else text
        def natural_keys(text):
            return [atoi(c) for c in re.split(r'(\d+)', text)]

        image_paths = os.listdir(f'{self.model_name}/animation')
        image_paths.sort(key=natural_keys)

        if len(image_paths) > 0:
            images = []
            for path in image_paths:
                img = Image.open(f'{self.model_name}/animation/{path}').convert('RGBA')
                images.append(img)

            # Save images as a GIF
            imageio.mimwrite(f"{self.model_name}/plots/prediction_process.gif", images, duration=15)

    def save_predictions(self):
        if not os.path.exists(f"{self.model_name}/predictions"):
            os.makedirs(f"{self.model_name}/predictions")

        points_to_predict = self.geometry.random_points(100000)
        prediction_for_these_points = self.model.predict(points_to_predict)

        with open(f"{self.model_name}/predictions/domain_points.npy", 'wb') as f:
            np.save(f, points_to_predict)

        with open(f"{self.model_name}/predictions/prediction_for_points.npy", 'wb') as f:
            np.save(f, prediction_for_these_points)