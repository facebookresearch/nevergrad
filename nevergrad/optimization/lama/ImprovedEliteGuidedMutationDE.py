import numpy as np


class ImprovedEliteGuidedMutationDE:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.pop_size = 40
        self.initial_mutation_factor = 0.8
        self.final_mutation_factor = 0.3
        self.crossover_prob = 0.8
        self.elitism_rate = 0.2
        self.archive_size = 50
        self.stagnation_threshold = 25

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
        archive = []

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
            archive.extend(new_pop)
            if len(archive) > self.archive_size:
                archive = archive[-self.archive_size :]

            if self.budget % 50 == 0 and archive:
                archive_idx = np.random.choice(len(archive))
                archive_ind = archive[archive_idx]
                archive_fitness = func(archive_ind)
                if archive_fitness < self.f_opt:
                    self.f_opt = archive_fitness
                    self.x_opt = archive_ind

            # Stagnation handling
            if best_fitness == self.f_opt:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
                best_fitness = self.f_opt

            if self.stagnation_counter >= self.stagnation_threshold:
                # Re-initialize part of the population
                reinited_pop = np.random.uniform(lower_bound, upper_bound, (self.pop_size // 2, self.dim))
                reinited_fitness = np.array([func(ind) for ind in reinited_pop])
                self.budget -= self.pop_size // 2

                pop = np.vstack((elite_pop, new_pop[elite_count : self.pop_size // 2]))
                fitness = np.hstack((elite_fitness, fitness[elite_count : self.pop_size // 2]))

                combined_pop = np.vstack((pop, reinited_pop))
                combined_fitness = np.hstack((fitness, reinited_fitness))

                pop = combined_pop
                fitness = combined_fitness

                self.stagnation_counter = 0

            new_pop = np.array(new_pop)
            combined_pop = np.vstack((elite_pop, new_pop[elite_count:]))
            combined_fitness = np.hstack((elite_fitness, fitness[elite_count:]))

            pop = combined_pop
            fitness = combined_fitness

            generation += 1

        return self.f_opt, self.x_opt
