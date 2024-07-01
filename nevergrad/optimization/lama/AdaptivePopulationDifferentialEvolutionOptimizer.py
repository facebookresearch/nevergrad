import numpy as np


class AdaptivePopulationDifferentialEvolutionOptimizer:
    def __init__(self, budget=10000, pop_size=50, init_F=0.8, init_CR=0.9):
        self.budget = budget
        self.pop_size = pop_size
        self.init_F = init_F
        self.init_CR = init_CR
        self.dim = 5  # As stated, dimensionality is 5
        self.bounds = (-5.0, 5.0)  # Bounds are given as [-5.0, 5.0]

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.eval_count = self.pop_size

        # Differential weights and crossover probabilities for each individual
        F_values = np.full(self.pop_size, self.init_F)
        CR_values = np.full(self.pop_size, self.init_CR)

        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                # Mutation
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                F = F_values[i]
                mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])

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
                    fitness[i] = f_trial
                    population[i] = trial
                    # Self-adapting parameters
                    F_values[i] = F * 1.1 if F < 1 else F
                    CR_values[i] = CR * 1.1 if CR < 1 else CR
                else:
                    F_values[i] = F * 0.9 if F > 0 else F
                    CR_values[i] = CR * 0.9 if CR > 0 else CR

                if self.eval_count >= self.budget:
                    break

            # Population adaptation
            if self.eval_count % (self.budget // 10) == 0:
                mean_fitness = np.mean(fitness)
                std_dev_fitness = np.std(fitness)
                new_pop_size = int(self.pop_size * (1 + std_dev_fitness / mean_fitness))
                new_pop_size = min(
                    max(new_pop_size, 10), 100
                )  # Keep population size within reasonable limits

                if new_pop_size != self.pop_size:
                    if new_pop_size > self.pop_size:
                        new_individuals = np.random.uniform(
                            self.bounds[0], self.bounds[1], (new_pop_size - self.pop_size, self.dim)
                        )
                        new_fitness = np.array([func(ind) for ind in new_individuals])
                        self.eval_count += new_fitness.size
                        population = np.concatenate((population, new_individuals))
                        fitness = np.concatenate((fitness, new_fitness))
                        F_values = np.concatenate((F_values, np.full(new_individuals.shape[0], self.init_F)))
                        CR_values = np.concatenate(
                            (CR_values, np.full(new_individuals.shape[0], self.init_CR))
                        )
                    else:
                        selected_indices = np.argsort(fitness)[:new_pop_size]
                        population = population[selected_indices]
                        fitness = fitness[selected_indices]
                        F_values = F_values[selected_indices]
                        CR_values = CR_values[selected_indices]

                    self.pop_size = new_pop_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        return self.f_opt, self.x_opt
