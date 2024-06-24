import numpy as np


class AdvancedRAMEDSv6:
    def __init__(
        self,
        budget,
        population_size=50,
        crossover_rate=0.92,
        F_min=0.5,
        F_max=0.9,
        memory_size=50,
        elite_size=10,
    ):
        self.budget = budget
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.F_min = F_min
        self.F_max = F_max
        self.memory_size = memory_size
        self.elite_size = elite_size
        self.lb, self.ub, self.dimension = -5.0, 5.0, 5

    def __call__(self, func):
        # Initialize population and fitness
        population = self.lb + (self.ub - self.lb) * np.random.rand(self.population_size, self.dimension)
        fitness = np.array([func(individual) for individual in population])

        # Initialize memory and elite structures
        memory = np.empty((self.memory_size, self.dimension))
        memory_fitness = np.full(self.memory_size, np.inf)
        elite = np.empty((self.elite_size, self.dimension))
        elite_fitness = np.full(self.elite_size, np.inf)

        # Best solution tracking
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        performance_switch_threshold = 0.1
        use_random_mutation = False
        last_improvement = 0

        while evaluations < self.budget:
            # Update elites
            elite_indices = np.argsort(fitness)[: self.elite_size]
            elite = population[elite_indices].copy()
            elite_fitness = fitness[elite_indices].copy()

            for i in range(self.population_size):
                # Adaptive mutation factor with dynamic modulation
                F = self.F_min + (self.F_max - self.F_min) * np.sin(np.pi * evaluations / self.budget)

                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                if use_random_mutation or np.random.rand() < performance_switch_threshold:
                    mutant = np.clip(a + F * (b - c), self.lb, self.ub)
                else:
                    best_or_elite = (
                        best_solution
                        if np.random.rand() < 0.7
                        else elite[np.random.randint(0, self.elite_size)]
                    )
                    mutant = np.clip(
                        population[i] + F * (best_or_elite - population[i] + a - b), self.lb, self.ub
                    )

                # Crossover
                cross_points = np.random.rand(self.dimension) < self.crossover_rate
                trial = np.where(cross_points, mutant, population[i])

                # Selection and memory update
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    if (
                        evaluations - last_improvement > self.population_size * 2
                    ):  # Switch strategy if stagnant
                        use_random_mutation = not use_random_mutation
                        last_improvement = evaluations

                    worst_idx = np.argmax(memory_fitness)
                    if memory_fitness[worst_idx] > fitness[i]:
                        memory[worst_idx] = population[i].copy()
                        memory_fitness[worst_idx] = fitness[i]

                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
