import numpy as np


class QuantumEnhancedDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.initial_pop_size = 50
        self.tau1 = 0.1
        self.tau2 = 0.1
        self.elitism_rate = 0.2
        self.memory_size = 5
        self.memory_F = [0.8] * self.memory_size
        self.memory_CR = [0.9] * self.memory_size
        self.memory_index = 0
        self.min_pop_size = 30
        self.max_pop_size = 100
        self.phase_switch_ratio = 0.3
        self.local_search_iters = 5
        self.diversity_threshold = 1e-4

    def initialize_population(self, bounds, pop_size):
        return np.random.uniform(bounds.lb, bounds.ub, (pop_size, self.dim))

    def select_parents(self, population, idx, pop_size):
        indices = np.delete(np.arange(pop_size), idx)
        return population[np.random.choice(indices, 3, replace=False)]

    def mutate(self, base, diff1, diff2, F):
        return np.clip(base + F * (diff1 - diff2), -5.0, 5.0)

    def crossover(self, target, mutant, CR):
        j_rand = np.random.randint(self.dim)
        return np.where(np.random.rand(self.dim) < CR, mutant, target)

    def diversity(self, population):
        return np.mean(np.std(population, axis=0))

    def adapt_parameters(self):
        F = np.random.choice(self.memory_F)
        CR = np.random.choice(self.memory_CR)
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

    def restart_population(self, bounds, pop_size):
        return self.initialize_population(bounds, pop_size)

    def adaptive_population_size(self):
        return np.random.randint(self.min_pop_size, self.max_pop_size + 1)

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        bounds = func.bounds

        population = self.initialize_population(bounds, self.initial_pop_size)
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.initial_pop_size

        phase_switch_evals = int(self.phase_switch_ratio * self.budget)

        while evaluations < self.budget:
            new_population = np.zeros_like(population)
            new_fitness = np.zeros_like(fitness)

            for i in range(self.initial_pop_size):
                if evaluations >= self.budget:
                    break

                parents = self.select_parents(population, i, self.initial_pop_size)
                parent1, parent2, parent3 = parents

                F, CR = self.adapt_parameters()

                if evaluations < phase_switch_evals:
                    mutant = self.mutate(parent1, parent2, parent3, F)
                else:
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

            elite_indices = np.argsort(new_fitness)[: int(self.elitism_rate * self.initial_pop_size)]
            for idx in elite_indices:
                if evaluations >= self.budget:
                    break
                new_population[idx], new_fitness[idx] = self.local_search(new_population[idx], bounds, func)
                evaluations += self.local_search_iters

            if self.diversity(new_population) < self.diversity_threshold and evaluations < self.budget:
                new_pop_size = self.adaptive_population_size()
                new_population = self.restart_population(bounds, new_pop_size)
                new_fitness = np.array([func(ind) for ind in new_population])
                evaluations += new_pop_size
                self.initial_pop_size = new_pop_size

            population = new_population
            fitness = new_fitness

            self.memory_F[self.memory_index] = F
            self.memory_CR[self.memory_index] = CR
            self.memory_index = (self.memory_index + 1) % self.memory_size

        return self.f_opt, self.x_opt
