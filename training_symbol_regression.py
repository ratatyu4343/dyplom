import math

from kovasznay_flow_symbol_regression_model import *

X_DATA_PATH = "./models_statistic/model30_3_1000/predictions/domain_points.npy"
Y_DATA_PATH = "./models_statistic/model30_3_1000/predictions/prediction_for_points.npy"

BIN_OPERATORS = ["+", "*", "-", "pow"]
UNAR_OPERATORS = ["cos", "sin", "exp", "sqrt"]

model100_22 = KovasznayFlowSymbolRegressionModel(
    model_name="model100_22",
    X_file=X_DATA_PATH,
    y_file=Y_DATA_PATH,
    binary_operators=BIN_OPERATORS,
    unary_operators=UNAR_OPERATORS,
    npoints=100,
    niterations=100,
    populations=5,
    population_size=22
)

model100_55 = KovasznayFlowSymbolRegressionModel(
    model_name="model100_55",
    X_file=X_DATA_PATH,
    y_file=Y_DATA_PATH,
    binary_operators=BIN_OPERATORS,
    unary_operators=UNAR_OPERATORS,
    npoints=100,
    niterations=100,
    populations=5,
    population_size=55
)

model100_100 = KovasznayFlowSymbolRegressionModel(
    model_name="model100_100",
    X_file=X_DATA_PATH,
    y_file=Y_DATA_PATH,
    binary_operators=BIN_OPERATORS,
    unary_operators=UNAR_OPERATORS,
    npoints=100,
    niterations=100,
    populations=5,
    population_size=100
)

model100_22.train()
model100_22.save_results()

model100_55.train()
model100_55.save_results()

model100_100.train()
model100_100.save_results()


'''
model1000_22 = KovasznayFlowSymbolRegressionModel(
    model_name="model1000_22",
    X_file=X_DATA_PATH,
    y_file=Y_DATA_PATH,
    binary_operators=BIN_OPERATORS,
    unary_operators=UNAR_OPERATORS,
    npoints=1000,
    niterations=100,
    populations=5,
    population_size=22
)

model10000_22 = KovasznayFlowSymbolRegressionModel(
    model_name="model10000_22",
    X_file=X_DATA_PATH,
    y_file=Y_DATA_PATH,
    binary_operators=BIN_OPERATORS,
    unary_operators=UNAR_OPERATORS,
    npoints=10000,
    niterations=100,
    populations=5,
    population_size=22
)
'''
