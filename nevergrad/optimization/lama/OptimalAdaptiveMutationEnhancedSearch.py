import numpy as np


class OptimalAdaptiveMutationEnhancedSearch:
    def __init__(
        self,
        budget,
        population_size=50,
        initial_crossover_rate=0.9,
        F_min=0.2,
        F_max=1.0,
        memory_size=50,
        elite_size=10,
    ):
        self.budget = budget
        self.population_size = population_size
        self.crossover_rate = initial_crossover_rate
        self.F_min = F_min
        self.F_max = F_max
        self.memory_size = memory_size
        self.elite_size = elite_size

    def __call__(self, func):
        lb, ub, dimension = -5.0, 5.0, 5

        # Initialize population and fitness
        population = lb + (ub - lb) * np.random.rand(self.population_size, dimension)
        fitness = np.array([func(individual) for individual in population])

        # Initialize memory for good solutions and their fitness
        memory = np.empty((self.memory_size, dimension))
        memory_fitness = np.full(self.memory_size, np.inf)

        # Initialize elite solutions and their fitness
        elite = np.empty((self.elite_size, dimension))
        elite_fitness = np.full(self.elite_size, np.inf)

        # Best solution tracking
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        while evaluations < self.budget:
            # Update elite
            sorted_indices = np.argsort(fitness)
            elite = population[sorted_indices[: self.elite_size]]
            elite_fitness = fitness[sorted_indices[: self.elite_size]]

            # Adaptive mutation based on fitness deviation and time-progress
            F = self.F_min + (self.F_max - self.F_min) * np.exp(-1.0 * np.var(fitness) / np.ptp(fitness))

            for i in range(self.population_size):
                # Selection of mutation strategy based on adaptive rates
                if np.random.rand() < 0.5:
                    mutation_strategy = "rand"
                    idxs = np.random.choice(
                        [idx for idx in range(self.population_size) if idx != i], 3, replace=False
                    )
                    a, b, c = population[idxs]
                    mutant = a + F * (b - c)
                else:
                    mutation_strategy = "best"
                    a = population[np.random.choice([idx for idx in range(self.population_size) if idx != i])]
                    mutant = best_solution + F * (a - best_solution)

                mutant = np.clip(mutant, lb, ub)  # Ensure mutant is within bounds

                # Crossover
                cross_points = np.random.rand(dimension) < self.crossover_rate
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    # Update the memory with good solutions
                    worse_memory_idx = np.argmax(memory_fitness)
                    if fitness[i] < memory_fitness[worse_memory_idx]:
                        memory[worse_memory_idx] = population[i]
                        memory_fitness[worse_memory_idx] = fitness[i]

                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
