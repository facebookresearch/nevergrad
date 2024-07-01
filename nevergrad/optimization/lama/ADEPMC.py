import numpy as np


class ADEPMC:
    def __init__(
        self,
        budget,
        dimension=5,
        lower_bound=-5.0,
        upper_bound=5.0,
        population_size=50,
        F_base=0.5,
        CR_base=0.5,
        learning_rate=0.1,
    ):
        self.budget = budget
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population_size = population_size
        self.F_base = F_base  # Base mutation factor
        self.CR_base = CR_base  # Base crossover probability
        self.learning_rate = learning_rate  # Learning rate for adaptive parameters

    def __call__(self, func):
        # Initialize population and fitness
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dimension)
        )
        fitness = np.array([func(ind) for ind in population])
        best_index = np.argmin(fitness)
        best_individual = population[best_index]
        best_fitness = fitness[best_index]
        evaluations = self.population_size

        # Adaptive mutation and crossover probabilities
        F_adaptive = np.full(self.population_size, self.F_base)
        CR_adaptive = np.full(self.population_size, self.CR_base)

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Mutation
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = a + F_adaptive[i] * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                trial = np.where(np.random.rand(self.dimension) < CR_adaptive[i], mutant, population[i])
                trial_fitness = func(trial)
                evaluations += 1

                # Selection and adaptivity update
                if trial_fitness < fitness[i]:
                    population[i], fitness[i] = trial, trial_fitness
                    F_adaptive[i] += self.learning_rate * (1.0 - F_adaptive[i])  # Increase mutation factor
                    CR_adaptive[i] -= self.learning_rate * CR_adaptive[i]  # Decrease crossover probability
                    if trial_fitness < best_fitness:
                        best_fitness, best_individual = trial_fitness, trial.copy()
                else:
                    F_adaptive[i] -= self.learning_rate * F_adaptive[i]  # Decrease mutation factor
                    CR_adaptive[i] += self.learning_rate * (
                        1.0 - CR_adaptive[i]
                    )  # Increase crossover probability

        return best_fitness, best_individual
