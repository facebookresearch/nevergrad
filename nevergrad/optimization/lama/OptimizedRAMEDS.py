import numpy as np


class OptimizedRAMEDS:
    def __init__(
        self,
        budget,
        population_size=50,
        crossover_rate=0.9,
        F_min=0.1,
        F_max=0.9,
        memory_size=20,
        elite_size=5,
    ):
        self.budget = budget
        self.population_size = population_size
        self.crossover_rate = crossover_rate
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
            # Update elites
            elite_indices = np.argsort(fitness)[: self.elite_size]
            elite = population[elite_indices].copy()
            elite_fitness = fitness[elite_indices].copy()

            for i in range(self.population_size):
                # Dynamic Mutation Factor
                adaptive_F = self.F_max - (evaluations / self.budget) * (self.F_max - self.F_min)

                # Mutation strategy with diversity consideration
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                best_or_elite = (
                    best_solution if np.random.rand() < 0.5 else elite[np.random.randint(0, self.elite_size)]
                )
                mutant = np.clip(population[i] + adaptive_F * (best_or_elite - population[i] + a - b), lb, ub)

                # Crossover with diversity enhancement
                trial = np.where(np.random.rand(dimension) < self.crossover_rate, mutant, population[i])
                trial_fitness = func(trial)
                evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update memory with diversity strategy
                    if trial_fitness < np.max(memory_fitness):
                        update_idx = np.argmax(memory_fitness)
                        memory[update_idx] = trial
                        memory_fitness[update_idx] = trial_fitness

                    # Update the best found solution
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
