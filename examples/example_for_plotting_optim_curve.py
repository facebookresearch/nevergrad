import nevergrad as ng

def f(x):
   return sum((x-1.234)**2)
opt = ng.optimizers.TwoPointsDE(3, 500)
opt.minimize(f)
print(opt.optim_curve)
