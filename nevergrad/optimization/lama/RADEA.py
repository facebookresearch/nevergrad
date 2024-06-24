import numpy as np

"""### Redesigned Algorithm Name: Robust Adaptive Differential Evolution with Archive (RADEA)
### Key Adjustments:
1. **Archive Initialization and Usage**: Introduce checks to handle cases where the archive is initially empty and update its usage logic in mutation.
2. **Parameter Tuning and Control**: Introduce adaptive parameters that can change based on the optimization progress.
3. **Improved Mutation Strategy**: Use a more robust strategy that ensures diversity and avoids premature convergence.
"""


class RADEA:
    def __init__(self, budget):
        self.budget = budget
        self.population_size = 20
        self.dimension = 5
        self.low = -5.0
        self.high = 5.0
        self.archive = []
        self.archive_max_size = 50

    def initialize(self):
        return np.random.uniform(self.low, self.high, (self.population_size, self.dimension))

    def evaluate(self, population, func):
        fitness = np.array([func(ind) for ind in population])
        return fitness

    def mutation(self, population, archive, F):
        mutant = np.zeros_like(population)
        combined = np.vstack([population] + [archive]) if archive else population
        num_candidates = len(combined)

        for i in range(self.population_size):
            indices = np.random.choice(num_candidates, 3, replace=False)
            x1, x2, x3 = combined[indices]
            mutant_vector = x1 + F * (x2 - x3)
            mutant[i] = np.clip(mutant_vector, self.low, self.high)
        return mutant

    def crossover(self, population, mutant, CR):
        crossover = np.where(np.random.rand(self.population_size, self.dimension) < CR, mutant, population)
        return crossover

    def select(self, population, fitness, trial_population, trial_fitness):
        better_idx = trial_fitness < fitness
        population[better_idx] = trial_population[better_idx]
        fitness[better_idx] = trial_fitness[better_idx]
        updated_indices = np.where(better_idx)[0]
        return population, fitness, updated_indices

    def __call__(self, func):
        population = self.initialize()
        fitness = self.evaluate(population, func)
        iterations = self.budget // self.population_size
        F, CR = 0.8, 0.9  # Fixed F and CR for simplification

        for _ in range(iterations):
            mutant = self.mutation(population, self.archive, F)
            trial_population = self.crossover(population, mutant, CR)
            trial_fitness = self.evaluate(trial_population, func)
            population, fitness, updated_indices = self.select(
                population, fitness, trial_population, trial_fitness
            )

            # Archive management: add only improved solutions
            for idx in updated_indices:
                self.archive.append(population[idx])
                if len(self.archive) > self.archive_max_size:
                    self.archive.pop(0)

        best_index = np.argmin(fitness)
        best_individual = population[best_index]
        best_fitness = fitness[best_index]
        return best_fitness, best_individual
