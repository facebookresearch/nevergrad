import numpy as np


class RefinedUltraRefinedRAMEDS:
    def __init__(
        self,
        budget,
        population_size=50,
        crossover_rate=0.95,
        F_min=0.3,
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

    def __call__(self, func):
        lb, ub, dimension = -5.0, 5.0, 5

        # Initialize population and fitness
        population = lb + (ub - lb) * np.random.rand(self.population_size, dimension)
        fitness = np.array([func(individual) for individual in population])

        # Initialize memory for good solutions and their fitness
        memory = np.empty((self.memory_size, dimension))
        memory_fitness = np.full(self.memory_size, np.inf)

        # Best solution tracking
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        while evaluations < self.budget:
            # Update mutation factor dynamically for precision as the optimization progresses
            progress = evaluations / self.budget
            F = self.F_max - (self.F_max - self.F_min) * progress

            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                # Introducing dynamic weighting for the best and random contribution
                best_weight = 0.5 + 0.5 * np.sin(np.pi * progress)  # Increases as progress goes
                random_weight = 1 - best_weight

                mutant = np.clip(
                    a + F * (best_weight * (best_solution - a) + random_weight * (b - c)), lb, ub
                )

                # Crossover with decreased likelihood towards the end
                cross_points = np.random.rand(dimension) < (self.crossover_rate * (1 - progress))
                trial = np.where(cross_points, mutant, population[i])

                # Evaluation and selection
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
