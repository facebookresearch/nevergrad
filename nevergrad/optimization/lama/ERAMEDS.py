import numpy as np


class ERAMEDS:
    def __init__(
        self,
        budget,
        population_size=50,
        initial_crossover_rate=0.95,
        F_min=0.5,
        F_max=0.9,
        memory_size=50,
        elite_size=10,
        memory_fade_interval=100,
    ):
        self.budget = budget
        self.population_size = population_size
        self.initial_crossover_rate = initial_crossover_rate
        self.crossover_rate = initial_crossover_rate
        self.F_min = F_min
        self.F_max = F_max
        self.memory_size = memory_size
        self.elite_size = elite_size
        self.memory_fade_interval = memory_fade_interval

    def __call__(self, func):
        lb, ub, dimension = -5.0, 5.0, 5

        # Initialize population and fitness
        population = lb + (ub - lb) * np.random.rand(self.population_size, dimension)
        fitness = np.array([func(individual) for individual in population])

        # Initialize memory for good solutions and their fitness
        memory = np.empty((self.memory_size, dimension))
        memory_fitness = np.full(self.memory_size, np.inf)
        memory_age = np.zeros(self.memory_size, dtype=int)

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

            # Dynamically adjust crossover rate
            self.crossover_rate = self.initial_crossover_rate - evaluations / self.budget * (
                self.initial_crossover_rate - 0.6
            )

            for i in range(self.population_size):
                # Adaptive mutation factor with dynamic modulation
                F = self.F_max - (self.F_max - self.F_min) * np.cos(np.pi * evaluations / self.budget)

                # Mutation: DE/current-to-best/1 with enhanced selection
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                use_elite = np.random.rand() < 0.6 + 0.4 * (best_fitness / (elite_fitness.mean() + 1e-6))
                best_or_elite = elite[np.random.randint(0, self.elite_size)] if use_elite else best_solution
                mutant = np.clip(population[i] + F * (best_or_elite - population[i] + a - b), lb, ub)

                # Crossover
                cross_points = np.random.rand(dimension) < self.crossover_rate
                trial = np.where(cross_points, mutant, population[i])

                # Selection and updating memory with fading mechanism
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    # Memory update with fading mechanism
                    worst_idx = np.argmax(memory_fitness)
                    if (
                        memory_fitness[worst_idx] > fitness[i]
                        or memory_age[worst_idx] > self.memory_fade_interval
                    ):
                        memory[worst_idx] = population[i].copy()
                        memory_fitness[worst_idx] = fitness[i]
                        memory_age[worst_idx] = 0
                    else:
                        memory_age[worst_idx] += 1

                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
