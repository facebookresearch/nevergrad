import numpy as np


class AdaptiveLearningDifferentialEvolutionOptimizer:
    def __init__(self, budget=10000, pop_size=50, init_F=0.8, init_CR=0.9):
        self.budget = budget
        self.pop_size = pop_size
        self.init_F = init_F
        self.init_CR = init_CR
        self.dim = 5  # Dimensionality is 5
        self.bounds = (-5.0, 5.0)  # Bounds are [-5.0, 5.0]

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.eval_count = self.pop_size

        # Differential weights and crossover probabilities for each individual
        F_values = np.full(self.pop_size, self.init_F)
        CR_values = np.full(self.pop_size, self.init_CR)

        # Initialize archive to store successful mutation vectors
        archive = []

        while self.eval_count < self.budget:
            new_population = []
            new_fitness = []
            for i in range(self.pop_size):
                # Mutation with archive usage
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                if archive:
                    d = archive[np.random.randint(len(archive))]
                    mutant = np.clip(
                        a + F_values[i] * (b - c) + F_values[i] * (a - d), self.bounds[0], self.bounds[1]
                    )
                else:
                    mutant = np.clip(a + F_values[i] * (b - c), self.bounds[0], self.bounds[1])

                # Crossover
                CR = CR_values[i]
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                f_trial = func(trial)
                self.eval_count += 1
                if f_trial < fitness[i]:
                    new_population.append(trial)
                    new_fitness.append(f_trial)
                    archive.append(population[i])
                    # Limit archive size
                    if len(archive) > self.pop_size:
                        archive.pop(np.random.randint(len(archive)))
                    # Self-adapting parameters
                    F_values[i] = F_values[i] * 1.1 if F_values[i] < 1 else F_values[i]
                    CR_values[i] = CR_values[i] * 1.1 if CR_values[i] < 1 else CR_values[i]
                else:
                    new_population.append(population[i])
                    new_fitness.append(fitness[i])
                    F_values[i] = F_values[i] * 0.9 if F_values[i] > 0 else F_values[i]
                    CR_values[i] = CR_values[i] * 0.9 if CR_values[i] > 0 else CR_values[i]

                if self.eval_count >= self.budget:
                    break

            # Replace the old population with the new one
            population = np.array(new_population)
            fitness = np.array(new_fitness)

            # Learning Phase: Adjust F and CR based on the success rate
            success_rate = np.count_nonzero(np.array(new_fitness) < np.array(fitness)) / self.pop_size
            if success_rate > 0.2:
                self.init_F = min(1.0, self.init_F * 1.1)
                self.init_CR = min(1.0, self.init_CR * 1.1)
            else:
                self.init_F = max(0.1, self.init_F * 0.9)
                self.init_CR = max(0.1, self.init_CR * 0.9)
            F_values = np.full(self.pop_size, self.init_F)
            CR_values = np.full(self.pop_size, self.init_CR)

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        return self.f_opt, self.x_opt
