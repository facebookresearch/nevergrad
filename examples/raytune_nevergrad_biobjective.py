import nevergrad as ng
import numpy as np
import nevergrad as ng
from ray import train, tune
from ray.tune.search.nevergrad import NevergradSearch

# Sample data
expected_returns_simulated = np.array([0.08, 0.12, 0.10, 0.15, 0.09])
covariance_matrix_simulated = np.array([[0.0004, 0.0003, 0.0002, 0.0001, 0.0002],
                                        [0.0003, 0.0009, 0.0004, 0.0002, 0.0003],
                                        [0.0002, 0.0004, 0.0010, 0.0003, 0.0004],
                                        [0.0001, 0.0002, 0.0003, 0.0008, 0.0002],
                                        [0.0002, 0.0003, 0.0004, 0.0002, 0.0005]])

# Bi-objective problem: Financial portfolio optimization
def portfolio_optimization(weights, expected_returns, covariance_matrix):
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_volatility = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
    # maximize portfolio return and minimize portfolio volatility
    return [-portfolio_return, portfolio_volatility]

# Using Nevergrad to solve the bi-objective problem
optimizer = ng.optimizers.CMA(parametrization= 5, budget=100)

initial_weights = np.random.dirichlet(np.ones(5), size=1).flatten()  

optimizer.tell(ng.p.Array(init=initial_weights.tolist()), [0, 1],
               portfolio_optimization(initial_weights, expected_returns_simulated, covariance_matrix_simulated))

# Minimize the bi-objective problem
optimizer.minimize(lambda x: portfolio_optimization(x, expected_returns_simulated, covariance_matrix_simulated), verbosity=0, constraint_violation=[lambda x: -np.sum(x) + 1, lambda x: np.min(x) - 0])

print("Pareto front:")
for param in sorted(optimizer.pareto_front(), key=lambda p: p.losses[0]):
    print(f"{param} with losses {param.losses}")

# Using Raytune with Nevergrad to solve the same problem
def evaluate(weight_1, weight_2, weight_3, weight_4, weight_5, alpha1, alpha2):
    weights = [weight_1, weight_2, weight_3, weight_4, weight_5]

    portfolio_return = sum(weights[i] * expected_returns_simulated[i] for i in range(5))

    portfolio_volatility = sum(
        weights[i] * weights[j] * covariance_matrix_simulated[i, j] 
        for i in range(5) for j in range(5)
    )

    portfolio_volatility = np.sqrt(portfolio_volatility)
    
    return -alpha1 * portfolio_return + alpha2 * portfolio_volatility

def objective(config):
    alpha1 = config["alpha1"]
    alpha2 = 1 - alpha1
    config["alpha2"] = alpha2

    weights = [config["weight_1"], config["weight_2"], config["weight_3"], config["weight_4"], config["weight_5"]]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    config['weight_1'] = normalized_weights[0]
    config['weight_2'] = normalized_weights[1]
    config['weight_3'] = normalized_weights[2]
    config['weight_4'] = normalized_weights[3]
    config['weight_5'] = normalized_weights[4]

    score = evaluate(config["weight_1"], config["weight_2"], config["weight_3"], config["weight_4"], config["weight_5"], alpha1, alpha2)
    train.report({"mean_loss": score})

algo = NevergradSearch(
    optimizer=ng.optimizers.CMA,
)

algo = tune.search.ConcurrencyLimiter(algo, max_concurrent=4)

num_samples = 100

search_config = {
    "weight_1": tune.uniform(0, 1),
    "weight_2": tune.uniform(0, 1),
    "weight_3": tune.uniform(0, 1),
    "weight_4": tune.uniform(0, 1),
    "weight_5": tune.uniform(0, 1),
    "alpha1": tune.uniform(0, 1),
    "alpha2": 0
}

tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        metric="mean_loss",
        mode="min",
        search_alg=algo,
        num_samples=num_samples,
    ),
    param_space=search_config,
)
results = tuner.fit()

best_result_config = results.get_best_result().config
formatted_config = {key: round(value, 2) for key, value in best_result_config.items()}
print("Best results found were: ", formatted_config)