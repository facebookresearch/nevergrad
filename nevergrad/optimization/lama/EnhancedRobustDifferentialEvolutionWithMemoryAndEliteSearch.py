import numpy as np


class EnhancedRobustDifferentialEvolutionWithMemoryAndEliteSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.pop_size = 50
        self.initial_F = 0.8
        self.initial_CR = 0.9
        self.tau1 = 0.1
        self.tau2 = 0.1
        self.elitism_rate = 0.2
        self.memory_size = 5
        self.memory_F = [self.initial_F] * self.memory_size
        self.memory_CR = [self.initial_CR] * self.memory_size
        self.memory_index = 0
        self.diversity_threshold = 1e-5
        self.learning_rate = 0.1

        # For adaptive population sizing
        self.min_pop_size = 20
        self.max_pop_size = 70

    def initialize_population(self, bounds, pop_size):
        return np.random.uniform(bounds.lb, bounds.ub, (pop_size, self.dim))

    def select_parents(self, population, idx):
        indices = np.delete(np.arange(len(population)), idx)
        return population[np.random.choice(indices, 3, replace=False)]

    def mutate(self, parent1, parent2, parent3, F):
        return np.clip(parent1 + F * (parent2 - parent3), -5.0, 5.0)

    def crossover(self, target, mutant, CR):
        j_rand = np.random.randint(self.dim)
        return np.where(np.random.rand(self.dim) < CR, mutant, target)

    def diversity(self, population):
        return np.mean(np.std(population, axis=0))

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
        for _ in range(5):
            mutation = np.random.randn(self.dim) * 0.01
            trial = np.clip(individual + mutation, bounds.lb, bounds.ub)
            trial_fitness = func(trial)
            if trial_fitness < best_fitness:
                best_individual = trial
                best_fitness = trial_fitness
        return best_individual, best_fitness

    def elite_learning(self, elite, global_best):
        return np.clip(
            elite + self.learning_rate * np.random.randn(self.dim) * (global_best - elite), -5.0, 5.0
        )

    def restart_population(self, bounds, pop_size):
        return self.initialize_population(bounds, pop_size)

    def adaptive_population_size(self, current_pop_size):
        if np.random.rand() < 0.5:
            new_pop_size = int(current_pop_size * 1.1)
        else:
            new_pop_size = int(current_pop_size * 0.9)
        return np.clip(new_pop_size, self.min_pop_size, self.max_pop_size)

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        bounds = func.bounds

        population = self.initialize_population(bounds, self.pop_size)
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        evaluations = self.pop_size

        while evaluations < self.budget:
            new_population = np.zeros((self.pop_size, self.dim))
            fitness = np.zeros(self.pop_size)

            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                parents = self.select_parents(population, i)
                parent1, parent2, parent3 = parents
                F, CR = self.adapt_parameters()
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

            elite_indices = np.argsort(fitness)[: int(self.elitism_rate * self.pop_size)]
            for i in elite_indices:
                if evaluations >= self.budget:
                    break
                new_population[i], fitness[i] = self.local_search(new_population[i], bounds, func)
                evaluations += 5

            elite_population = new_population[elite_indices]
            non_elite_indices = np.argsort(fitness)[int(self.elitism_rate * self.pop_size) :]
            for i in non_elite_indices:
                if evaluations >= self.budget:
                    break
                learned_trial = self.elite_learning(new_population[i], global_best_position)
                learned_fitness = func(learned_trial)
                evaluations += 1

                if learned_fitness < fitness[i]:
                    new_population[i] = learned_trial
                    fitness[i] = learned_fitness
                    if learned_fitness < self.f_opt:
                        self.f_opt = learned_fitness
                        self.x_opt = learned_trial

            if self.diversity(new_population) < self.diversity_threshold and evaluations < self.budget:
                new_pop_size = self.adaptive_population_size(self.pop_size)  # Adapt population size here
                new_population = self.restart_population(bounds, new_pop_size)
                personal_best_positions = np.copy(new_population)
                personal_best_scores = np.array([func(ind) for ind in new_population])
                global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
                global_best_score = np.min(personal_best_scores)
                evaluations += new_pop_size
                self.pop_size = new_pop_size  # Update population size

            population = np.copy(new_population)

            self.memory_F[self.memory_index] = F
            self.memory_CR[self.memory_index] = CR
            self.memory_index = (self.memory_index + 1) % self.memory_size

        return self.f_opt, self.x_opt
