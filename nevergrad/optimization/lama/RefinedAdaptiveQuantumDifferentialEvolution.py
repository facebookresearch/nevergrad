import numpy as np


class RefinedAdaptiveQuantumDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.pop_size = 50
        self.initial_F = 0.8
        self.initial_CR = 0.9
        self.tau1 = 0.1
        self.tau2 = 0.1
        self.alpha = 0.1  # Scale for quantum jumps
        self.local_search_budget = 5

    def initialize_population(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))

    def select_parents(self, population, idx):
        indices = list(range(0, idx)) + list(range(idx + 1, self.pop_size))
        idxs = np.random.choice(indices, 3, replace=False)
        return population[idxs]

    def mutate(self, parent1, parent2, parent3, F):
        return np.clip(parent1 + F * (parent2 - parent3), -5.0, 5.0)

    def crossover(self, target, mutant, CR):
        j_rand = np.random.randint(self.dim)
        return np.where(np.random.rand(self.dim) < CR, mutant, target)

    def diversity(self, population):
        return np.mean(np.std(population, axis=0))

    def adapt_parameters(self, F, CR):
        if np.random.rand() < self.tau1:
            F = np.clip(np.random.normal(F, 0.1), 0, 1)
        if np.random.rand() < self.tau2:
            CR = np.clip(np.random.normal(CR, 0.1), 0, 1)
        return F, CR

    def local_search(self, individual, bounds, func):
        for _ in range(self.local_search_budget):
            mutation = np.random.randn(self.dim) * 0.01
            trial = np.clip(individual + mutation, bounds.lb, bounds.ub)
            trial_fitness = func(trial)
            if trial_fitness < func(individual):
                individual = trial
        return individual

    def quantum_jump(self, individual, global_best, alpha):
        return np.clip(individual + alpha * np.random.randn(self.dim) * (global_best - individual), -5.0, 5.0)

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        bounds = func.bounds

        population = self.initialize_population(bounds)
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        evaluations = self.pop_size

        F = self.initial_F
        CR = self.initial_CR

        while evaluations < self.budget:
            new_population = np.zeros((self.pop_size, self.dim))
            fitness = np.zeros(self.pop_size)

            for i in range(self.pop_size):
                parents = self.select_parents(population, i)
                parent1, parent2, parent3 = parents
                F, CR = self.adapt_parameters(F, CR)
                mutant = self.mutate(parent1, parent2, parent3, F)
                trial = self.crossover(population[i], mutant, CR)

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < personal_best_scores[i]:
                    personal_best_positions[i] = trial
                    personal_best_scores[i] = trial_fitness

                if personal_best_scores[i] < global_best_score:
                    global_best_position = personal_best_positions[i]
                    global_best_score = personal_best_scores[i]

                new_population[i] = trial
                fitness[i] = trial_fitness

                if evaluations >= self.budget:
                    break

            if np.min(fitness) < self.f_opt:
                self.f_opt = np.min(fitness)
                self.x_opt = population[np.argmin(fitness)]

            elite_indices = np.argsort(fitness)[: self.pop_size // 2]
            elite_population = new_population[elite_indices]

            for i in range(len(elite_indices)):
                elite_population[i] = self.local_search(elite_population[i], bounds, func)
                evaluations += self.local_search_budget

            if evaluations < self.budget:
                population = np.copy(new_population)
                for i in range(self.pop_size):
                    if np.random.rand() < 0.5:
                        parents = self.select_parents(population, i)
                        parent1, parent2, parent3 = parents
                        mutant = self.mutate(global_best_position, parent1, parent2, F)
                        trial = self.crossover(population[i], mutant, CR)
                        trial_fitness = func(trial)
                        evaluations += 1

                        if trial_fitness < fitness[i]:
                            population[i] = trial
                            fitness[i] = trial_fitness
                            if trial_fitness < self.f_opt:
                                self.f_opt = trial_fitness
                                self.x_opt = trial
                    else:
                        quantum_trial = self.quantum_jump(population[i], global_best_position, self.alpha)
                        quantum_fitness = func(quantum_trial)
                        evaluations += 1

                        if quantum_fitness < fitness[i]:
                            population[i] = quantum_trial
                            fitness[i] = quantum_fitness
                            if quantum_fitness < self.f_opt:
                                self.f_opt = quantum_fitness
                                self.x_opt = quantum_trial
                    if evaluations >= self.budget:
                        break

        return self.f_opt, self.x_opt
