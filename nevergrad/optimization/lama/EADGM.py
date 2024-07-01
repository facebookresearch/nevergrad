import numpy as np


class EADGM:
    def __init__(
        self,
        budget,
        dimension=5,
        lower_bound=-5.0,
        upper_bound=5.0,
        population_size=200,
        F_base=0.6,
        CR_base=0.9,
        learning_rate=0.05,
        p_best=0.2,
    ):
        self.budget = budget
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population_size = population_size
        self.F_base = F_base  # Initial mutation factor
        self.CR_base = CR_base  # Initial crossover probability
        self.learning_rate = learning_rate  # Learning rate for adaptive parameters
        self.p_best = p_best  # Probability of using best individual updates

    def __call__(self, func):
        # Initialize population and fitness
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dimension)
        )
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        best_index = np.argmin(fitness)
        best_individual = population[best_index]
        best_fitness = fitness[best_index]

        # Adaptive mutation and crossover probabilities
        F_adaptive = np.full(self.population_size, self.F_base)
        CR_adaptive = np.full(self.population_size, self.CR_base)

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Select indices for mutation, ensuring all are unique and not the current index
                indices = [idx for idx in range(self.population_size) if idx != i]
                selected_indices = np.random.choice(indices, 3, replace=False)
                a, b, c = population[selected_indices]

                # Mutation with Gaussian addition for enhanced exploration
                G = np.random.normal(0, 1, self.dimension)
                mutant = np.clip(a + F_adaptive[i] * (b - c) + G, self.lower_bound, self.upper_bound)

                # Crossover
                trial = np.where(np.random.rand(self.dimension) < CR_adaptive[i], mutant, population[i])
                trial_fitness = func(trial)
                evaluations += 1

                # Selection and adaptivity update
                if trial_fitness < fitness[i]:
                    population[i], fitness[i] = trial, trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness, best_individual = trial_fitness, trial.copy()
                    # Adaptive factor update towards successful mutations
                    F_adaptive[i] += self.learning_rate * (1.0 - F_adaptive[i])
                    CR_adaptive[i] -= self.learning_rate * CR_adaptive[i]
                else:
                    # Adaptive factor degradation towards unsuccessful mutations
                    F_adaptive[i] -= self.learning_rate * F_adaptive[i]
                    CR_adaptive[i] += self.learning_rate * (1.0 - CR_adaptive[i])

        return best_fitness, best_individual
