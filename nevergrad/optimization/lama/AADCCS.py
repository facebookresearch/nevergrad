import numpy as np


class AADCCS:
    def __init__(
        self,
        budget,
        dimension=5,
        lower_bound=-5.0,
        upper_bound=5.0,
        population_size=150,
        F_base=0.5,
        CR_base=0.8,
        learning_rate=0.1,
        p=0.25,
    ):
        self.budget = budget
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population_size = population_size
        self.F_base = F_base  # Initial mutation factor
        self.CR_base = CR_base  # Initial crossover probability
        self.learning_rate = learning_rate  # Learning rate for adaptive parameters
        self.p = p  # Probability of using best individual updates

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

                # Choose different indices for mutation, ensuring all are unique
                indices = np.random.choice(self.population_size, 4, replace=False)
                a, b, c, d = population[indices]

                # Mutation with best individual influence
                if np.random.rand() < self.p:
                    a = best_individual  # Using best individual to guide mutation

                # Differential mutation and crossover
                mutant = np.clip(a + F_adaptive[i] * ((b - c) + (a - d)), self.lower_bound, self.upper_bound)
                trial = np.where(np.random.rand(self.dimension) < CR_adaptive[i], mutant, population[i])
                trial_fitness = func(trial)
                evaluations += 1

                # Selection and adaptivity update
                if trial_fitness < fitness[i]:
                    population[i], fitness[i] = trial, trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness, best_individual = trial_fitness, trial.copy()
                    # Adaptive factor update towards successful mutations
                    F_adaptive[i] = max(0.1, F_adaptive[i] + self.learning_rate * (1.0 - F_adaptive[i]))
                    CR_adaptive[i] = min(1.0, CR_adaptive[i] - self.learning_rate * CR_adaptive[i])
                else:
                    # Adaptive factor degradation towards unsuccessful mutations
                    F_adaptive[i] = max(0.1, F_adaptive[i] - self.learning_rate * F_adaptive[i])
                    CR_adaptive[i] = min(1.0, CR_adaptive[i] + self.learning_rate * (1.0 - CR_adaptive[i]))

        return best_fitness, best_individual
