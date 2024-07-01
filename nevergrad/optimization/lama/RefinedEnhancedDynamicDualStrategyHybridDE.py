import numpy as np


class RefinedEnhancedDynamicDualStrategyHybridDE:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.initial_pop_size = 60
        self.final_pop_size = 20
        self.initial_mutation_factor = 0.8
        self.final_mutation_factor = 0.3
        self.initial_crossover_prob = 0.9
        self.final_crossover_prob = 0.6
        self.elitism_rate = 0.3
        self.local_search_prob = 0.3
        self.archive = []
        self.tol = 1e-6  # Tolerance for convergence check

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        lower_bound = -5.0
        upper_bound = 5.0

        # Initialize population
        pop_size = self.initial_pop_size
        pop = np.random.uniform(lower_bound, upper_bound, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        self.budget -= pop_size

        generation = 0
        last_best_fitness = self.f_opt

        while self.budget > 0:
            # Adaptive mutation factor and crossover probability
            mutation_factor = self.initial_mutation_factor - (
                (self.initial_mutation_factor - self.final_mutation_factor)
                * (generation / (self.budget / self.initial_pop_size))
            )
            crossover_prob = self.initial_crossover_prob - (
                (self.initial_crossover_prob - self.final_crossover_prob)
                * (generation / (self.budget / self.initial_pop_size))
            )

            # Adaptive population size
            pop_size = max(
                self.final_pop_size,
                int(
                    self.initial_pop_size
                    - (self.initial_pop_size - self.final_pop_size)
                    * (generation / (self.budget / self.initial_pop_size))
                ),
            )
            elite_count = max(1, int(self.elitism_rate * pop_size))

            # Elitism: preserve top individuals
            elite_indices = np.argsort(fitness)[:elite_count]
            elite_pop = pop[elite_indices]
            elite_fitness = fitness[elite_indices]

            # Dual-strategy evolution
            new_pop = []
            for i in range(pop_size):
                if self.budget <= 0:
                    break

                if np.random.rand() < 0.5:
                    idxs = np.random.choice(range(elite_count), 3, replace=False)
                    x1, x2, x3 = elite_pop[idxs]
                else:
                    idxs = np.random.choice(range(pop_size), 3, replace=False)
                    x1, x2, x3 = pop[idxs]

                mutant = x1 + mutation_factor * (x2 - x3)
                mutant = np.clip(mutant, lower_bound, upper_bound)

                cross_points = np.random.rand(self.dim) < crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Introduce elitist guidance in crossover stage
                elite_target_idx = np.random.randint(elite_count)
                trial = trial + np.random.rand(self.dim) * (elite_pop[elite_target_idx] - trial)
                trial = np.clip(trial, lower_bound, upper_bound)

                # Local search phase with some probability
                if np.random.rand() < self.local_search_prob:
                    trial = self.local_search(trial, func)

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
            if len(self.archive) > self.initial_pop_size:
                self.archive = self.archive[-self.initial_pop_size :]

            if self.budget % int(pop_size * 0.1) == 0 and self.archive:
                archive_idx = np.random.choice(len(self.archive))
                archive_ind = self.archive[archive_idx]
                f_archive = func(archive_ind)
                self.budget -= 1
                if f_archive < self.f_opt:
                    self.f_opt = f_archive
                    self.x_opt = archive_ind

            new_pop = np.array(new_pop)
            combined_pop = np.vstack((elite_pop, new_pop[elite_count:]))
            combined_fitness = np.hstack(
                (elite_fitness, fitness[elite_count:] if len(fitness) > elite_count else np.array([]))
            )

            pop = combined_pop[:pop_size]  # Ensure pop size consistency
            fitness = combined_fitness[:pop_size]  # Ensure fitness size consistency

            # Convergence check
            if np.abs(self.f_opt - last_best_fitness) < self.tol:
                break  # Stop if the improvement is below the tolerance level
            last_best_fitness = self.f_opt

            generation += 1

        return self.f_opt, self.x_opt

    def local_search(self, x, func):
        best_x = x.copy()
        best_f = func(x)
        perturbation = 0.01 * (np.random.rand(self.dim) - 0.5)  # Smaller perturbation for finer adjustments
        new_x = x + perturbation
        new_x = np.clip(new_x, -5.0, 5.0)
        new_f = func(new_x)
        self.budget -= 1  # Account for the local search function evaluation
        if new_f < best_f:
            best_x = new_x
            best_f = new_f
        return best_x
