import numpy as np


class EDAG:
    def __init__(
        self,
        budget,
        population_size=100,
        initial_step=0.5,
        step_decay=0.95,
        differential_weight=0.6,
        crossover_prob=0.7,
    ):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0
        self.initial_step = initial_step
        self.step_decay = step_decay
        self.differential_weight = differential_weight
        self.crossover_prob = crossover_prob

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        num_evals = self.population_size

        step_size = self.initial_step
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()

        while num_evals < self.budget:
            new_population = np.zeros_like(population)

            # Generate new candidates
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)

                # Differential mutation
                mutant = population[a] + self.differential_weight * (population[b] - population[c])
                mutant = np.clip(mutant, self.lb, self.ub)

                # Crossover
                cross_points = np.random.rand(self.dimension) < self.crossover_prob
                trial = np.where(cross_points, mutant, population[i])

                # Adaptive gradient-based step
                gradient_direction = best_individual - population[i]
                trial += step_size * gradient_direction
                trial = np.clip(trial, self.lb, self.ub)

                trial_fitness = func(trial)
                num_evals += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial.copy()
                else:
                    new_population[i] = population[i]

            population = new_population
            step_size *= self.step_decay  # Adaptive decay of the step size

        return best_fitness, best_individual
