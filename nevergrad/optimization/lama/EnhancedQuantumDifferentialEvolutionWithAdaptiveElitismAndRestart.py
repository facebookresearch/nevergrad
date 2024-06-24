import numpy as np
from scipy.optimize import minimize


class EnhancedQuantumDifferentialEvolutionWithAdaptiveElitismAndRestart:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.bounds = (-5.0, 5.0)
        self.population_size = 60
        self.elite_size = 10
        self.alpha = 0.6
        self.beta = 0.5
        self.local_search_prob = 0.5
        self.CR = 0.9
        self.F = 0.8
        self.diversity_threshold = 1e-3
        self.adaptive_restart_interval = 50
        self.memory_update_interval = 25

    def random_bounds(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], self.dim)

    def local_search(self, x, func):
        result = minimize(func, x, method="L-BFGS-B", bounds=[(self.bounds[0], self.bounds[1])] * self.dim)
        return result.x, result.fun

    def quantum_update(self, x, elites):
        p_best = elites[np.random.randint(len(elites))]
        u = np.random.uniform(0, 1, self.dim)
        v = np.random.uniform(-1, 1, self.dim)
        Q = self.beta * (p_best - x) * np.log(1 / u)
        return np.clip(x + Q * v, self.bounds[0], self.bounds[1])

    def adaptive_restart(self, population, fitness, func):
        std_dev = np.std(fitness)
        if std_dev < self.diversity_threshold:
            population = np.array([self.random_bounds() for _ in range(self.population_size)])
            fitness = np.array([func(ind) for ind in population])
        return population, fitness

    def update_memory(self, memory, population, fitness):
        combined = sorted(list(memory) + list(zip(population, fitness)), key=lambda x: x[1])
        return combined[: self.elite_size]

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        population = np.array([self.random_bounds() for _ in range(self.population_size)])
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        memory = [(population[i], fitness[i]) for i in range(self.elite_size)]

        while evaluations < self.budget:
            for i in range(self.population_size):
                a, b, c = population[
                    np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                ]
                mutant = np.clip(a + self.F * (b - c), self.bounds[0], self.bounds[1])

                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                f_trial = func(trial)
                evaluations += 1

                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                if np.random.rand() < self.local_search_prob and evaluations < self.budget:
                    refined_trial, f_refined_trial = self.local_search(trial, func)
                    evaluations += 1
                    if f_refined_trial < fitness[i]:
                        population[i] = refined_trial
                        fitness[i] = f_refined_trial
                        if f_refined_trial < self.f_opt:
                            self.f_opt = f_refined_trial
                            self.x_opt = refined_trial

            memory = self.update_memory(memory, population, fitness)
            elite_particles = np.array([mem[0] for mem in memory])

            for i in range(self.population_size):
                trial = self.quantum_update(population[i], elite_particles)
                f_trial = func(trial)
                evaluations += 1

                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

            if evaluations % self.adaptive_restart_interval == 0:
                population, fitness = self.adaptive_restart(population, fitness, func)

            if evaluations % self.memory_update_interval == 0:
                memory = self.update_memory(memory, population, fitness)

        return self.f_opt, self.x_opt
