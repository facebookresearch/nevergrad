import numpy as np


class DualPopulationADE:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.pop_size = 40
        self.mutation_factor = 0.5
        self.crossover_prob = 0.7
        self.archive = []

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        lower_bound = -5.0
        upper_bound = 5.0

        # Initialize populations
        pop = np.random.uniform(lower_bound, upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        self.budget -= self.pop_size

        while self.budget > 0:
            new_pop = []
            for i in range(self.pop_size):
                if self.budget <= 0:
                    break

                # Mutation
                idxs = np.random.choice(range(self.pop_size), 3, replace=False)
                x1, x2, x3 = pop[idxs]
                mutant = x1 + self.mutation_factor * (x2 - x3)
                mutant = np.clip(mutant, lower_bound, upper_bound)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Selection
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
                if self.budget % 100 == 0 and self.archive:
                    archive_idx = np.random.choice(len(self.archive))
                    archive_ind = self.archive[archive_idx]
                    if func(archive_ind) < self.f_opt:
                        self.f_opt = func(archive_ind)
                        self.x_opt = archive_ind

            # Update archive
            self.archive.extend(new_pop)
            if len(self.archive) > self.pop_size:
                self.archive = self.archive[-self.pop_size :]

            pop = np.array(new_pop)

        return self.f_opt, self.x_opt
