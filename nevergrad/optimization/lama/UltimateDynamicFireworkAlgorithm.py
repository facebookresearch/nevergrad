import numpy as np


class UltimateDynamicFireworkAlgorithm:
    def __init__(
        self,
        population_size=50,
        max_sparks=10,
        max_generations=1500,
        initial_alpha=0.1,
        initial_beta=0.2,
        p_ex=0.9,
        p_dt=0.05,
        exploration_range=0.6,
        mutation_rate=0.20,
    ):
        self.population_size = population_size
        self.max_sparks = max_sparks
        self.max_generations = max_generations
        self.initial_alpha = initial_alpha
        self.initial_beta = initial_beta
        self.p_ex = p_ex
        self.p_dt = p_dt
        self.exploration_range = exploration_range
        self.mutation_rate = mutation_rate
        self.budget = 0
        self.f_opt = np.Inf
        self.x_opt = None

    def initialize_population(self, func):
        self.dim = func.bounds.ub.shape[0]
        self.population = np.random.uniform(
            func.bounds.lb, func.bounds.ub, size=(self.population_size, self.dim)
        )
        self.fireworks = [(np.copy(x), 0, np.Inf) for x in self.population]
        self.best_individual = None
        self.best_fitness = np.Inf
        self.alpha = np.full(self.population_size, self.initial_alpha)
        self.beta = np.full(self.population_size, self.initial_beta)

    def explosion_operator(self, x, beta):
        return x + np.random.uniform(-self.exploration_range, self.exploration_range, size=self.dim) * beta

    def attraction_operator(self, x, y, alpha):
        return x + alpha * (y - x)

    def mutation_operator(self, x):
        return x + np.random.normal(0, self.mutation_rate, size=self.dim)

    def update_parameters(self, k, fitness_diff):
        if fitness_diff < 0:
            self.alpha[k] *= 0.95  # Decrease alpha
            self.beta[k] *= 1.05  # Increase beta

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.initialize_population(func)

        for _ in range(self.max_generations):
            for i, (x, _, _) in enumerate(self.fireworks):
                fitness = func(x)
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_individual = np.copy(x)

                for _ in range(self.max_sparks):
                    if np.random.rand() < self.p_ex:
                        new_spark = self.explosion_operator(x, self.beta[i])
                    else:
                        j = np.random.randint(0, self.population_size)
                        new_spark = self.attraction_operator(x, self.fireworks[j][0], self.alpha[i])

                    new_spark = self.mutation_operator(new_spark)
                    new_fitness = func(new_spark)
                    fitness_diff = new_fitness - func(self.fireworks[i][0])
                    if fitness_diff < 0:
                        self.fireworks[i] = (np.copy(new_spark), 0, new_fitness)
                    else:
                        self.fireworks[i] = (
                            np.copy(self.fireworks[i][0]),
                            self.fireworks[i][1] + 1,
                            self.fireworks[i][2],
                        )

                    self.update_parameters(i, fitness_diff)

                if self.fireworks[i][1] > self.p_dt * self.max_sparks:
                    self.fireworks[i] = (
                        np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim),
                        0,
                        np.Inf,
                    )

        self.f_opt = func(self.best_individual)
        self.x_opt = self.best_individual

        return self.f_opt, self.x_opt
