import numpy as np


class AdaptiveElitistMutationDE:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.pop_size = 40
        self.initial_mutation_factor = 0.8
        self.final_mutation_factor = 0.3
        self.crossover_prob = 0.8
        self.elitism_rate = 0.2
        self.learning_rate = 0.1

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

            # Incorporate elitist mutation
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

            # Combine new population with elites
            new_pop = np.array(new_pop)
            combined_pop = np.vstack((elite_pop, new_pop[elite_count:]))
            combined_fitness = np.hstack((elite_fitness, fitness[elite_count:]))

            pop = combined_pop
            fitness = combined_fitness

            # Learning step: Adjust population towards elites
            for i in range(self.pop_size):
                if np.random.rand() < self.learning_rate:
                    closest_elite_idx = np.argmin(np.linalg.norm(elite_pop - pop[i], axis=1))
                    pop[i] += (elite_pop[closest_elite_idx] - pop[i]) * np.random.rand()

            generation += 1

        return self.f_opt, self.x_opt
