import itertools
import time

from kovasznay_flow_model import *


# Set global parameters
NEURON1 = 10
NEURON2 = 30
NEURON3 = 60

SHAPE1_1 = [2] + 1 * [NEURON1] + [3]
SHAPE2_1 = [2] + 1 * [NEURON2] + [3]
SHAPE3_1 = [2] + 1 * [NEURON3] + [3]
SHAPE1_2 = [2] + 2 * [NEURON1] + [3]
SHAPE2_2 = [2] + 2 * [NEURON2] + [3]
SHAPE3_2 = [2] + 2 * [NEURON3] + [3]
SHAPE1_3 = [2] + 3 * [NEURON1] + [3]
SHAPE2_3 = [2] + 3 * [NEURON2] + [3]
SHAPE3_3 = [2] + 3 * [NEURON3] + [3]

DOMAINPOINTS1 = 100
BOUNDARYPOINTS1 = 100
DOMAINPOINTS2 = 1000
BOUNDARYPOINTS2 = 1000
DOMAINPOINTS3 = 10000
BOUNDARYPOINTS3 = 10000

TESTPOINTS = 1000
SHOWPOINTS = 40000
ITERATIONS = 25000

COOLDOWNTIME = 60 * 2

# Set search geometry
geometry = dde.geometry.Rectangle(xmin=[-0.5, -0.5], xmax=[1, 1.5])

# Create combination matrix
row1 = [SHAPE1_1, SHAPE2_1, SHAPE3_1, SHAPE1_2, SHAPE2_2, SHAPE3_2, SHAPE1_3, SHAPE2_3, SHAPE3_3]
row2 = [[DOMAINPOINTS1, BOUNDARYPOINTS1], [DOMAINPOINTS2, BOUNDARYPOINTS2], [DOMAINPOINTS3, BOUNDARYPOINTS3]]
combinations = list(itertools.product(row1, row2))

# Set models
models = []
for combination in combinations:
    model = KovasznayFlowModel(
        model_name=f"model{combination[0][1]}_{len(combination[0])-2}_{combination[1][0]}",
        num_domain_points=combination[1][0],
        num_boundary_points=combination[1][1],
        num_test_points=TESTPOINTS,
        num_show_points=SHOWPOINTS,
        network_structure=combination[0],
        geometry=geometry,
        num_iterations=ITERATIONS,
        save_animation=True
    )
    models.append(model)

for model in models:
    model.train_model()
    time.sleep(COOLDOWNTIME)

