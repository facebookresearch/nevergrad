import numpy as np


class DynamicAdaptiveFireworkAlgorithm:
    def __init__(
        self,
        population_size=30,
        max_sparks=5,
        max_generations=1000,
        initial_alpha=0.1,
        initial_beta=0.2,
        p_ex=0.8,
        p_dt=0.1,
        mutation_rate=0.05,
    ):
        self.population_size = population_size
        self.max_sparks = max_sparks
        self.max_generations = max_generations
        self.initial_alpha = initial_alpha
        self.initial_beta = initial_beta
        self.p_ex = p_ex
        self.p_dt = p_dt
        self.mutation_rate = mutation_rate
        self.budget = 0
        self.f_opt = np.Inf
        self.x_opt = None

    def initialize_population(self, func):
        self.dim = func.bounds.ub.shape[0]
        self.population = np.random.uniform(
            func.bounds.lb, func.bounds.ub, size=(self.population_size, self.dim)
        )
        self.fireworks = [(np.copy(x), 0) for x in self.population]
        self.best_individual = None
        self.best_fitness = np.Inf
        self.alpha = np.full(self.population_size, self.initial_alpha)
        self.beta = np.full(self.population_size, self.initial_beta)

    def explosion_operator(self, x, func, beta):
        return x + np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim) * beta

    def attraction_operator(self, x, y, alpha):
        return x + alpha * (y - x)

    def distance(self, x, y):
        return np.linalg.norm(x - y)

    def update_parameters(self, k, fitness_diff):
        if fitness_diff < 0:
            self.alpha[k] *= 0.95  # Decrease alpha
            self.beta[k] *= 1.05  # Increase beta
        else:
            self.alpha[k] *= 1.05  # Increase alpha
            self.beta[k] *= 0.95  # Decrease beta

    def adapt_mutation_rate(self, fitness_diff):
        if fitness_diff < 0:
            self.mutation_rate *= 0.9  # Decrease mutation rate
        else:
            self.mutation_rate *= 1.1  # Increase mutation rate

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.initialize_population(func)

        for _ in range(self.max_generations):
            for i, (x, _) in enumerate(self.fireworks):
                fitness = func(x)
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_individual = np.copy(x)

                for _ in range(self.max_sparks):
                    if np.random.rand() < self.p_ex:
                        new_spark = self.explosion_operator(x, func, self.beta[i])
                    else:
                        j = np.random.randint(0, self.population_size)
                        new_spark = self.attraction_operator(x, self.fireworks[j][0], self.alpha[i])

                    new_spark += np.random.normal(0, self.mutation_rate, size=self.dim)
                    new_fitness = func(new_spark)

                    fitness_diff = new_fitness - func(self.fireworks[i][0])
                    if fitness_diff < 0:
                        self.fireworks[i] = (np.copy(new_spark), 0)
                    else:
                        self.fireworks[i] = (np.copy(self.fireworks[i][0]), self.fireworks[i][1] + 1)

                    self.update_parameters(i, fitness_diff)
                    self.adapt_mutation_rate(fitness_diff)

                if self.fireworks[i][1] > self.p_dt * self.max_sparks:
                    self.fireworks[i] = (np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim), 0)

        self.f_opt = func(self.best_individual)
        self.x_opt = self.best_individual

        return self.f_opt, self.x_opt
