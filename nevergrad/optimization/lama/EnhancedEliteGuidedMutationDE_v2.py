import numpy as np


class EnhancedEliteGuidedMutationDE_v2:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.pop_size = 40
        self.initial_mutation_factor = 0.8
        self.final_mutation_factor = 0.3
        self.crossover_prob = 0.8
        self.elitism_rate = 0.2
        self.archive = []
        self.stagnation_threshold = 20
        self.local_search_prob = 0.3

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        lower_bound = -5.0
        upper_bound = 5.0

        # Initialize population
        pop = np.random.uniform(lower_bound, upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        self.budget -= self.pop_size

        generation = 0
        best_fitness = self.f_opt
        self.stagnation_counter = 0

        while self.budget > 0:
            # Adaptive mutation factor
            mutation_factor = self.initial_mutation_factor - (
                (self.initial_mutation_factor - self.final_mutation_factor)
                * (generation / (self.budget / self.pop_size))
            )

            # Elitism: preserve top individuals
            elite_count = max(1, int(self.elitism_rate * self.pop_size))
            elite_indices = np.argsort(fitness)[:elite_count]
            elite_pop = pop[elite_indices]
            elite_fitness = fitness[elite_indices]

            # Incorporate elite-guided mutation
            new_pop = []
            for i in range(self.pop_size):
                if self.budget <= 0:
                    break

                if np.random.rand() < 0.5:
                    idxs = np.random.choice(range(self.pop_size), 3, replace=False)
                    x1, x2, x3 = pop[idxs]
                else:
                    idxs = np.random.choice(elite_count, 3, replace=False)
                    x1, x2, x3 = elite_pop[idxs]

                mutant = x1 + mutation_factor * (x2 - x3)
                mutant = np.clip(mutant, lower_bound, upper_bound)

                cross_points = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                f_trial = func(trial)
                self.budget -= 1
                if f_trial < fitness[i]:
                    new_pop.append(trial)
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    new_pop.append(pop[i])

            # Archive mechanism
            self.archive.extend(new_pop)
            if len(self.archive) > self.pop_size:
                self.archive = self.archive[-self.pop_size :]

            if self.budget % 50 == 0 and self.archive:
                archive_idx = np.random.choice(len(self.archive))
                archive_ind = self.archive[archive_idx]
                if func(archive_ind) < self.f_opt:
                    self.f_opt = func(archive_ind)
                    self.x_opt = archive_ind

            # Stagnation handling
            if best_fitness == self.f_opt:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
                best_fitness = self.f_opt

            if self.stagnation_counter >= self.stagnation_threshold:
                new_pop = np.random.uniform(lower_bound, upper_bound, (self.pop_size, self.dim))
                fitness = np.array([func(ind) for ind in new_pop])
                self.budget -= self.pop_size
                self.stagnation_counter = 0

            # Local search on elite individuals
            if np.random.rand() < self.local_search_prob and elite_count > 0:
                for j in range(elite_count):
                    perturbed = elite_pop[j] + np.random.normal(0, 0.01, self.dim)
                    perturbed = np.clip(perturbed, lower_bound, upper_bound)
                    f_perturbed = func(perturbed)
                    self.budget -= 1
                    if f_perturbed < elite_fitness[j]:
                        elite_pop[j] = perturbed
                        elite_fitness[j] = f_perturbed
                        if f_perturbed < self.f_opt:
                            self.f_opt = f_perturbed
                            self.x_opt = perturbed

            new_pop = np.array(new_pop)
            combined_pop = np.vstack((elite_pop, new_pop[elite_count:]))
            combined_fitness = np.hstack((elite_fitness, fitness[elite_count:]))

            pop = combined_pop
            fitness = combined_fitness

            generation += 1

        return self.f_opt, self.x_opt
