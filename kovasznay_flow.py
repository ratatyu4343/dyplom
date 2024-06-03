import os

os.environ["DDE_BACKEND"] = "tensorflow"

import matplotlib.pyplot as plt
from custom_callback import SavePredictionsCallback
from oda_equation import *
import deepxde as dde
import numpy as np

def boundary_outflow_condition(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1)

spatial_domain = dde.geometry.Rectangle(xmin=[-0.5, -0.5], xmax=[1, 1.5])

boundary_condition_u = dde.icbc.DirichletBC(
    spatial_domain, u_function, lambda _, on_boundary: on_boundary, component=0
)
boundary_condition_v = dde.icbc.DirichletBC(
    spatial_domain, v_function, lambda _, on_boundary: on_boundary, component=1
)
boundary_condition_right_p = dde.icbc.DirichletBC(
    spatial_domain, p_function, boundary_outflow_condition, component=2
)

data = dde.data.PDE(
    spatial_domain,
    pde_equations,
    [boundary_condition_u, boundary_condition_v, boundary_condition_right_p],
    num_domain=3000,
    num_boundary=400,
    num_test=1000,
)

net = dde.nn.FNN([2] + 2 * [30] + [3], "tanh", "Glorot normal")

model = dde.Model(data, net)

model.compile("adam", lr=1e-3)

imgs_list = []
X = spatial_domain.random_points(10000)
custom_callback = SavePredictionsCallback(100, X, imgs_list)

losshistory, train_state = model.train(iterations=10000, display_every=100, callbacks=[custom_callback])
dde.saveplot(losshistory, train_state, output_dir="./model_info/")


#model.compile("L-BFGS")
#losshistory, train_state = model.train()

output = model.predict(X)
u_pred = output[:, 0]
v_pred = output[:, 1]
p_pred = output[:, 2]

u_exact = u_function(X).reshape(-1)
v_exact = v_function(X).reshape(-1)
p_exact = p_function(X).reshape(-1)

f = model.predict(X, operator=pde_equations)

l2_difference_u = dde.metrics.l2_relative_error(u_exact, u_pred)
l2_difference_v = dde.metrics.l2_relative_error(v_exact, v_pred)
l2_difference_p = dde.metrics.l2_relative_error(p_exact, p_pred)
residual = np.mean(np.absolute(f))

print("Mean residual:", residual)
print("L2 relative error in u:", l2_difference_u)
print("L2 relative error in v:", l2_difference_v)
print("L2 relative error in p:", l2_difference_p)

# Розрахунок скалярної швидкості для передбаченого та аналітичного розв'язків
velocity_magnitude_pred = np.sqrt(u_pred**2 + v_pred**2)
velocity_magnitude_exact = np.sqrt(u_exact**2 + v_exact**2)

# Виведення скалярної швидкості
plt.figure(figsize=(10, 5))
plt.scatter(X[:, 0], X[:, 1], c=velocity_magnitude_pred, cmap='viridis')
plt.colorbar(label='Predicted velocity magnitude')
plt.title('Predicted velocity magnitude')
plt.savefig("./model_info/prediction_vel.png")

plt.figure(figsize=(10, 5))
plt.scatter(X[:, 0], X[:, 1], c=velocity_magnitude_exact, cmap='viridis')
plt.colorbar(label='Analytical velocity magnitude')
plt.title('Analytical velocity magnitude')
plt.savefig("./model_info/analitic_vel.png")

# Обчислення тиску для передбачених та аналітичних значень
plt.figure(figsize=(10, 5))
plt.scatter(X[:, 0], X[:, 1], c=p_pred, cmap='viridis')
plt.colorbar(label='Predicted pressure')
plt.title('Predicted pressure')
plt.savefig("./model_info/prediction_pres.png")

plt.figure(figsize=(10, 5))
plt.scatter(X[:, 0], X[:, 1], c=p_exact, cmap='viridis')
plt.colorbar(label='Analytical pressure')
plt.title('Analytical pressure')
plt.savefig("./model_info/analitic_pres.png")

# Обчислення модуля різниці між передбаченими та аналітичними значеннями швидкості та тиску
velocity_difference = np.abs(velocity_magnitude_pred - velocity_magnitude_exact)
pressure_difference = np.abs(p_pred - p_exact)

# Виведення модуля різниці
plt.figure(figsize=(10, 5))
plt.scatter(X[:, 0], X[:, 1], c=velocity_difference, cmap='binary')
plt.colorbar(label='Velocity difference (absolute)')
plt.title('Absolute velocity difference')
plt.savefig("./model_info/eror_vel.png")

plt.figure(figsize=(10, 5))
plt.scatter(X[:, 0], X[:, 1], c=pressure_difference,  cmap='binary')
plt.colorbar(label='Pressure difference')
plt.title('Pressure difference')
plt.savefig("./model_info/eror_pres.png")

for i in range(len(imgs_list)):
    u_pred = imgs_list[i][:, 0]
    v_pred = imgs_list[i][:, 1]
    velocity_magnitude_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)
    try:
        plt.figure(figsize=(10, 5))
        plt.scatter(X[:, 0], X[:, 1], c=velocity_magnitude_pred, cmap='viridis')
        plt.colorbar(label='Predicted velocity magnitude')
        plt.title('Predicted velocity magnitude')
        plt.savefig(f"./model_info/animation/prediction{i}.png")
    finally:
        plt.close()