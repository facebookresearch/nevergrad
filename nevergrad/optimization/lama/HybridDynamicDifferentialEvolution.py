import numpy as np


class HybridDynamicDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.dim = 5
        self.population_size = 50
        self.initial_F = 0.8  # Differential weight
        self.initial_CR = 0.9  # Crossover probability
        self.elite_rate = 0.2  # Elite rate to maintain a portion of elites
        self.eval_count = 0
        self.local_search_rate = 0.1  # Probability for local search

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_index = np.argmin(fitness)
        best_position = population[best_index]
        best_value = fitness[best_index]

        self.eval_count = self.population_size

        def local_search(position):
            # Simple local search strategy
            step_size = 0.1
            candidate = position + np.random.uniform(-step_size, step_size, position.shape)
            candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
            return candidate

        def adapt_parameters():
            # Self-adaptive strategy for F and CR with random components
            adaptive_F = self.initial_F + (0.1 * np.random.rand() - 0.05)
            adaptive_CR = self.initial_CR + (0.1 * np.random.rand() - 0.05)
            return np.clip(adaptive_F, 0.5, 1.0), np.clip(adaptive_CR, 0.5, 1.0)

        while self.eval_count < self.budget:
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)

            # Sort population by fitness and maintain elites
            elite_count = int(self.elite_rate * self.population_size)
            sorted_indices = np.argsort(fitness)
            elites = population[sorted_indices[:elite_count]]
            new_population[:elite_count] = elites
            new_fitness[:elite_count] = fitness[sorted_indices[:elite_count]]

            for i in range(elite_count, self.population_size):
                if self.eval_count >= self.budget:
                    break
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                F, CR = adapt_parameters()
                mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
                trial = np.copy(population[i])
                for d in range(self.dim):
                    if np.random.rand() < CR:
                        trial[d] = mutant[d]

                if np.random.rand() < self.local_search_rate:
                    candidate = local_search(trial)
                else:
                    candidate = trial

                candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                candidate_value = func(candidate)
                self.eval_count += 1

                if candidate_value < fitness[i]:
                    new_population[i] = candidate
                    new_fitness[i] = candidate_value

                if candidate_value < best_value:
                    best_value = candidate_value
                    best_position = candidate

            # Update population for the next iteration
            population = new_population
            fitness = new_fitness

        return best_value, best_position


# Example usage:
# func = SomeBlackBoxFunction()  # The black box function to be optimized
# optimizer = HybridDynamicDifferentialEvolution(budget=10000)
# best_value, best_position = optimizer(func)
