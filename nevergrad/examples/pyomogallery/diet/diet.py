import os
from pyomo.environ import *
import nevergrad as ng
from nevergrad.functions.pyomo.core import Pyomo
# The code is adopted from https://github.com/Pyomo/PyomoGallery/tree/master/diet

infinity = float('inf')

model = AbstractModel()

# Foods
model.F = Set()
# Nutrients
model.N = Set()

# Cost of each food
model.c    = Param(model.F, within=PositiveReals)
# Amount of nutrient in each food
model.a    = Param(model.F, model.N, within=NonNegativeReals)
# Lower and upper bound on each nutrient
model.Nmin = Param(model.N, within=NonNegativeReals, default=0.0)
model.Nmax = Param(model.N, within=NonNegativeReals, default=infinity)
# Volume per serving of food
model.V    = Param(model.F, within=PositiveReals)
# Maximum volume of food consumed
model.Vmax = Param(within=PositiveReals)

# Number of servings consumed of each food
model.x = Var(model.F, within=NonNegativeIntegers)

# Minimize the cost of food that is consumed
def cost_rule(model):
    return sum(model.c[i]*model.x[i] for i in model.F)
model.cost = Objective(rule=cost_rule)

# Limit nutrient consumption for each nutrient
def nutrient_rule(model, j):
    v = sum(model.a[i, j]*model.x[i] for i in model.F)
    return (model.Nmin[j], v, model.Nmax[j])
model.nutrient_limit = Constraint(model.N, rule=nutrient_rule)

# Limit the volume of food consumed
def volume_rule(model):
    return (None, sum(model.V[i]*model.x[i] for i in model.F), model.Vmax)
model.volume = Constraint(rule=volume_rule)

# Solve by Nevergrad
data = DataPortal()
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, "diet.dat")
data.load(filename=filename, model=model)
pyo_instance = model.create_instance(data)

func = Pyomo('Diet Model', pyo_instance)
optimizer = ng.optimizers.NGO(parametrization=func.parametrization, budget=100)
recommendation = optimizer.minimize(func.function)

for k, v in recommendation.kwargs.items():
    print(f"{k}: {v}")
