import numpy as np
from scipy.optimize import minimize


class EnhancedDynamicFireworkAlgorithmWithAdaptiveLocalSearchAndDynamicMutationV5:
    def __init__(
        self,
        population_size=30,
        max_sparks=5,
        max_generations=1000,
        initial_alpha=0.1,
        initial_beta=0.2,
        p_ex=0.8,
        p_dt=0.1,
        local_search_rate=0.2,
        local_search_trials=5,
        mutation_rate=0.1,
    ):
        self.population_size = population_size
        self.max_sparks = max_sparks
        self.max_generations = max_generations
        self.initial_alpha = initial_alpha
        self.initial_beta = initial_beta
        self.p_ex = p_ex
        self.p_dt = p_dt
        self.local_search_rate = local_search_rate
        self.local_search_trials = local_search_trials
        self.mutation_rate = mutation_rate
        self.budget = 0
        self.f_opt = np.Inf
        self.x_opt = None

    def initialize_population(self, func):
        self.dim = func.bounds.ub.shape[0]
        self.population = np.random.uniform(
            func.bounds.lb, func.bounds.ub, size=(self.population_size, self.dim)
        )
        self.fireworks = [(np.copy(x), 0, self.initial_alpha, self.initial_beta) for x in self.population]
        self.best_individual = None
        self.best_fitness = np.Inf

    def explosion_operator(self, x, func, beta):
        return x + np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim) * beta

    def attraction_operator(self, x, y, alpha):
        return x + alpha * (y - x)

    def distance(self, x, y):
        return np.linalg.norm(x - y)

    def update_parameters(self, k):
        self.fireworks[k] = self.fireworks[k][:2] + (self.fireworks[k][2] * 0.95, self.fireworks[k][3] * 1.05)

    def local_search(self, x, func):
        res = minimize(
            func,
            x,
            bounds=[(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)],
            options={"maxiter": self.local_search_trials},
        )
        return res.x

    def mutate(self, x, func):
        if np.random.rand() < self.mutation_rate:
            return np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
        return x

    def run_firework_algorithm(self, func):
        self.initialize_population(func)

        for _ in range(self.max_generations):
            for i, (x, _, alpha, beta) in enumerate(self.fireworks):
                fitness = func(x)
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_individual = np.copy(x)

                for _ in range(self.max_sparks):
                    if np.random.rand() < self.p_ex:
                        new_spark = self.explosion_operator(x, func, beta)
                    else:
                        j = np.random.randint(0, self.population_size)
                        new_spark = self.attraction_operator(x, self.fireworks[j][0], alpha)

                    if np.random.rand() < self.local_search_rate:
                        new_spark = self.local_search(new_spark, func)

                    new_spark = self.mutate(new_spark, func)

                    if func(new_spark) < func(self.fireworks[i][0]):
                        self.fireworks[i] = (np.copy(new_spark), 0, alpha, beta)
                        self.update_parameters(i)
                    else:
                        self.fireworks[i] = (
                            np.copy(self.fireworks[i][0]),
                            self.fireworks[i][1] + 1,
                            alpha,
                            beta,
                        )

                if self.fireworks[i][1] > self.p_dt * self.max_sparks:
                    self.fireworks[i] = (
                        np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim),
                        0,
                        self.initial_alpha,
                        self.initial_beta,
                    )

        self.f_opt = func(self.best_individual)
        self.x_opt = self.best_individual

    def __call__(self, func):
        self.run_firework_algorithm(func)

        return self.f_opt, self.x_opt
