import numpy as np


class QuantumLevyEliteMemeticDEHybridOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.pop_size = 50
        self.memory_size = 5
        self.memory_index = 0
        self.memory_F = [0.5] * self.memory_size
        self.memory_CR = [0.5] * self.memory_size
        self.tau1 = 0.1
        self.tau2 = 0.1
        self.local_search_iters = 5
        self.elitism_rate = 0.2
        self.diversity_threshold = 1e-4
        self.local_search_prob = 0.2
        self.alpha = 0.01

    def initialize_population(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))

    def select_parents(self, population, idx):
        indices = np.delete(np.arange(self.pop_size), idx)
        return population[np.random.choice(indices, 3, replace=False)]

    def mutate(self, base, diff1, diff2, F):
        return np.clip(base + F * (diff1 - diff2), -5.0, 5.0)

    def crossover(self, target, mutant, CR):
        j_rand = np.random.randint(self.dim)
        return np.where(np.random.rand(self.dim) < CR, mutant, target)

    def adapt_parameters(self):
        F = self.memory_F[self.memory_index]
        CR = self.memory_CR[self.memory_index]
        if np.random.rand() < self.tau1:
            F = np.clip(np.random.normal(F, 0.1), 0, 1)
        if np.random.rand() < self.tau2:
            CR = np.clip(np.random.normal(CR, 0.1), 0, 1)
        return F, CR

    def local_search(self, individual, bounds, func):
        best_individual = np.copy(individual)
        best_fitness = func(best_individual)
        for _ in range(self.local_search_iters):
            mutation = np.random.randn(self.dim) * 0.05
            trial = np.clip(individual + mutation, bounds.lb, bounds.ub)
            trial_fitness = func(trial)
            if trial_fitness < best_fitness:
                best_individual = trial
                best_fitness = trial_fitness
        return best_individual, best_fitness

    def levy_flight(self, individual, bounds):
        u = np.random.normal(0, 1, self.dim) * self.alpha
        v = np.random.normal(0, 1, self.dim)
        step = u / np.abs(v) ** (1 / 3)
        return np.clip(individual + step, bounds.lb, bounds.ub)

    def hybrid_local_search(self, individual, bounds, func):
        if np.random.rand() < self.local_search_prob:
            return self.local_search(individual, bounds, func)
        else:
            mutation = self.levy_flight(individual, bounds)
            trial_fitness = func(mutation)
            return (
                (mutation, trial_fitness)
                if trial_fitness < func(individual)
                else (individual, func(individual))
            )

    def diversity(self, population):
        return np.mean(np.std(population, axis=0))

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        bounds = func.bounds

        population = self.initialize_population(bounds)
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.pop_size

        while evaluations < self.budget:
            new_population = np.zeros_like(population)
            new_fitness = np.zeros_like(fitness)

            for i in range(self.pop_size):
                parents = self.select_parents(population, i)
                parent1, parent2, parent3 = parents

                F, CR = self.adapt_parameters()
                mutant = self.mutate(parent1, parent2, parent3, F)
                trial = self.crossover(population[i], mutant, CR)

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = trial_fitness
                else:
                    new_population[i] = population[i]
                    new_fitness[i] = fitness[i]

                if trial_fitness < self.f_opt:
                    self.f_opt = trial_fitness
                    self.x_opt = trial

            elite_indices = np.argsort(new_fitness)[: int(self.elitism_rate * self.pop_size)]
            for idx in elite_indices:
                new_population[idx], new_fitness[idx] = self.hybrid_local_search(
                    new_population[idx], bounds, func
                )
                evaluations += self.local_search_iters

            if self.diversity(new_population) < self.diversity_threshold and evaluations < self.budget:
                new_population = self.initialize_population(bounds)
                new_fitness = np.array([func(ind) for ind in new_population])
                evaluations += self.pop_size

            population = new_population
            fitness = new_fitness

            self.memory_F[self.memory_index] = F
            self.memory_CR[self.memory_index] = CR
            self.memory_index = (self.memory_index + 1) % self.memory_size

        return self.f_opt, self.x_opt
