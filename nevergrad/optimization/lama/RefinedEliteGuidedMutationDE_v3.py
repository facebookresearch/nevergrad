import numpy as np


class RefinedEliteGuidedMutationDE_v3:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.pop_size = 40
        self.initial_mutation_factor = 0.8
        self.final_mutation_factor = 0.3
        self.crossover_prob = 0.8
        self.elitism_rate = 0.2
        self.stagnation_threshold = 30

    def __call__(self, func):
        self.f_opt = np.inf
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
                * min(generation / (self.budget / self.pop_size), 1.0)
            )

            # Elitism: preserve top individuals
            elite_count = max(1, int(self.elitism_rate * self.pop_size))
            elite_indices = np.argsort(fitness)[:elite_count]
            elite_pop = pop[elite_indices]
            elite_fitness = fitness[elite_indices]

            new_pop = []
            for i in range(self.pop_size):
                if self.budget <= 0:
                    break

                if np.random.rand() < 0.5:
                    idxs = np.random.choice(range(self.pop_size), 3, replace=False)
                else:
                    idxs = np.random.choice(elite_count, 3, replace=False)

                x1, x2, x3 = pop[idxs[0]], pop[idxs[1]], pop[idxs[2]]
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

            pop = np.array(new_pop)
            fitness = np.array([func(ind) for ind in pop])
            self.budget -= self.pop_size

            if best_fitness == self.f_opt:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
                best_fitness = self.f_opt

            if self.stagnation_counter >= self.stagnation_threshold:
                reinit_count = self.pop_size // 2
                reinit_pop = np.random.uniform(lower_bound, upper_bound, (reinit_count, self.dim))
                reinit_fitness = np.array([func(ind) for ind in reinit_pop])
                self.budget -= reinit_count

                pop = np.vstack((elite_pop, reinit_pop))
                fitness = np.hstack((elite_fitness, reinit_fitness))

                self.stagnation_counter = 0

            generation += 1

        return self.f_opt, self.x_opt
