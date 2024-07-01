import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol
from sklearn.cluster import KMeans


class EnhancedClusteredDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality as per problem statement
        self.lb = -5.0
        self.ub = 5.0
        self.pop_size = 100
        self.num_clusters = 10
        self.F = 0.8
        self.CR = 0.9
        self.memory = []
        self.memory_size = 20
        self.elite_size = 5
        self.elite = []

    def _initialize_population(self):
        sobol_engine = Sobol(d=self.dim, scramble=False)
        sobol_samples = sobol_engine.random_base2(m=int(np.log2(self.pop_size // 2)))
        sobol_samples = self.lb + (self.ub - self.lb) * sobol_samples

        random_samples = np.random.uniform(self.lb, self.ub, (self.pop_size - len(sobol_samples), self.dim))
        return np.vstack((sobol_samples, random_samples))

    def _local_search(self, x, func):
        res = minimize(
            func, x, method="L-BFGS-B", bounds=[(self.lb, self.ub)] * self.dim, options={"disp": False}
        )
        return res.x, res.fun

    def _dynamic_parameters(self):
        self.F = np.random.uniform(0.4, 1.0)
        self.CR = np.random.uniform(0.1, 1.0)

    def _cluster_search(self, population, func):
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(population)
        cluster_centers = kmeans.cluster_centers_
        for center in cluster_centers:
            if self.evaluations >= self.budget:
                break
            local_opt, f_local_opt = self._local_search(center, func)
            self.evaluations += 1
            if f_local_opt < self.f_opt:
                self.f_opt = f_local_opt
                self.x_opt = local_opt

    def _memory_local_search(self, func):
        for mem in self.memory:
            if self.evaluations >= self.budget:
                break
            local_opt, f_local_opt = self._local_search(mem, func)
            self.evaluations += 1
            if f_local_opt < self.f_opt:
                self.f_opt = f_local_opt
                self.x_opt = local_opt

    def _adaptive_restart(self, population, fitness, func):
        mean_fitness = np.mean(fitness)
        std_fitness = np.std(fitness)
        if std_fitness < 1e-6 and self.evaluations < self.budget * 0.9:
            new_pop_size = min(self.pop_size * 2, self.budget - self.evaluations)
            new_population = np.random.uniform(self.lb, self.ub, (new_pop_size, self.dim))
            new_fitness = np.array([func(ind) for ind in new_population])
            self.evaluations += new_pop_size
            return new_population, new_fitness
        return population, fitness

    def _crossover(self, a, b, c):
        rand_idx = np.random.randint(self.dim)
        mutant_vector = np.copy(a)
        for j in range(self.dim):
            if np.random.rand() < self.CR or j == rand_idx:
                mutant_vector[j] = a[j] + self.F * (b[j] - c[j])
            else:
                mutant_vector[j] = a[j]
        return np.clip(mutant_vector, self.lb, self.ub)

    def _mutation_strategies(self, population):
        strategies = [
            lambda a, b, c: a + self.F * (b - c),
            lambda a, b, c: a + self.F * (b - np.mean(population, axis=0)),
            lambda a, b, c: a
            + self.F * (b - c)
            + self.F * (np.random.uniform(self.lb, self.ub, self.dim) - a),
        ]
        return strategies[np.random.randint(len(strategies))]

    def __call__(self, func):
        population = self._initialize_population()
        fitness = np.array([func(ind) for ind in population])

        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)].copy()
        self.evaluations = len(population)

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                if self.evaluations >= self.budget:
                    break
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutation_strategy = self._mutation_strategies(population)
                trial_vector = np.clip(a + mutation_strategy(a, b, c), self.lb, self.ub)
                f_candidate = func(trial_vector)
                self.evaluations += 1

                if f_candidate < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = f_candidate

                    if f_candidate < self.f_opt:
                        self.f_opt = f_candidate
                        self.x_opt = trial_vector

            # Update elite set
            elite_indices = np.argsort(fitness)[: self.elite_size]
            self.elite = [population[idx] for idx in elite_indices]

            if self.evaluations < self.budget:
                if len(self.memory) < self.memory_size:
                    self.memory.append(self.x_opt)
                else:
                    worst_mem_idx = np.argmax([func(mem) for mem in self.memory])
                    self.memory[worst_mem_idx] = self.x_opt

            self._dynamic_parameters()
            self._cluster_search(population, func)
            self._memory_local_search(func)
            population, fitness = self._adaptive_restart(population, fitness, func)

        return self.f_opt, self.x_opt
