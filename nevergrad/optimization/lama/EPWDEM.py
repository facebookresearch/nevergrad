import numpy as np


class EPWDEM:
    def __init__(
        self, budget, population_size=50, crossover_rate=0.85, F_base=0.6, F_amp=0.3, memory_factor=0.1
    ):
        self.budget = budget
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.F_base = F_base
        self.F_amp = F_amp
        self.memory_factor = memory_factor
        self.memory_size = int(memory_factor * population_size)

    def __call__(self, func):
        lb, ub, dimension = -5.0, 5.0, 5

        # Initialize population within the bounds
        population = lb + (ub - lb) * np.random.rand(self.population_size, dimension)
        fitness = np.array([func(individual) for individual in population])

        # Memory for elite solutions
        memory = np.empty((self.memory_size, dimension))
        memory_fitness = np.full(self.memory_size, np.inf)

        # Best solution tracking
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        while evaluations < self.budget:
            # Maintain a memory of best solutions
            sorted_indices = np.argsort(fitness)
            memory[: self.memory_size] = population[sorted_indices[: self.memory_size]]
            memory_fitness[: self.memory_size] = fitness[sorted_indices[: self.memory_size]]

            for i in range(self.population_size):
                # Adaptive mutation factor with progressive wave pattern
                F = self.F_base + self.F_amp * np.sin(2 * np.pi * evaluations / self.budget)

                # Mutation: Differential mutation with memory and best solution
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                memory_idx = np.argmin(memory_fitness)
                mutant = np.clip(memory[memory_idx] + F * (a - b), lb, ub)

                # Crossover: Exponential
                j = np.random.randint(dimension)
                trial = np.array(population[i])  # copy current to trial
                for k in range(dimension):
                    if np.random.rand() < self.crossover_rate or k == dimension - 1:
                        trial[j] = mutant[j]
                    j = (j + 1) % dimension  # modulo increment

                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
