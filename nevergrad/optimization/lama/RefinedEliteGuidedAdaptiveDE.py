import numpy as np


class RefinedEliteGuidedAdaptiveDE:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.pop_size = 40
        self.initial_mutation_factor = 0.8
        self.final_mutation_factor = 0.3
        self.crossover_prob = 0.8
        self.elitism_rate = 0.2
        self.archive_max_size = 100

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

            # Incorporate elite-guided mutation with better selection mechanism
            new_pop = []
            for i in range(self.pop_size):
                if self.budget <= 0:
                    break

                if np.random.rand() < 0.5:
                    idxs = np.random.choice(range(self.pop_size), 3, replace=False)
                    x1, x2, x3 = pop[idxs]
                else:
                    if archive and np.random.rand() < 0.5:
                        archive_idx = np.random.choice(len(archive), 3, replace=False)
                        x1, x2, x3 = np.array(archive)[archive_idx]
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
            if len(archive) > self.archive_max_size:
                archive = archive[-self.archive_max_size :]

            if self.budget % 50 == 0 and archive:
                archive_idx = np.random.choice(len(archive))
                archive_ind = archive[archive_idx]
                if func(archive_ind) < self.f_opt:
                    self.f_opt = func(archive_ind)
                    self.x_opt = archive_ind

            new_pop = np.array(new_pop)
            combined_pop = np.vstack((elite_pop, new_pop[: self.pop_size - elite_count]))
            combined_fitness = np.hstack((elite_fitness, fitness[: self.pop_size - elite_count]))

            pop = combined_pop
            fitness = combined_fitness

            generation += 1

        return self.f_opt, self.x_opt
