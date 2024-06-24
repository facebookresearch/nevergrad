import numpy as np


class EMSADE:
    def __init__(
        self, budget, population_size=100, F_base=0.5, CR_base=0.9, strategy_ratio=0.25, perturbation=0.1
    ):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0
        self.F_base = F_base  # Base mutation factor
        self.CR_base = CR_base  # Base crossover probability
        self.strategy_ratio = strategy_ratio  # Proportion of population applying each strategy
        self.perturbation = perturbation  # Perturbation for adaptive parameters

    def __call__(self, func):
        # Initialize population and fitness assessments
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dimension))
        fitness = np.array([func(x) for x in population])
        num_evals = self.population_size

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()

        while num_evals < self.budget:
            new_population = np.empty_like(population)

            for i in range(self.population_size):
                if num_evals >= self.budget:
                    break

                strategy_use = np.random.rand()
                F = self.F_base + self.perturbation * np.random.randn()
                CR = self.CR_base + self.perturbation * np.random.randn()

                if strategy_use < self.strategy_ratio:
                    # DE/current-to-best/1/bin
                    idxs = [idx for idx in range(self.population_size) if idx != i]
                    a, b = population[np.random.choice(idxs, 2, replace=False)]
                    mutant = population[i] + F * (best_individual - population[i]) + F * (a - b)
                else:
                    # DE/rand/2/bin
                    idxs = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c, d = population[np.random.choice(idxs, 4, replace=False)]
                    mutant = a + F * (b - c) + F * (c - d)

                mutant = np.clip(mutant, self.lb, self.ub)

                # Crossover
                cross_points = np.random.rand(self.dimension) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dimension)] = True
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                num_evals += 1

                # Selection
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial.copy()
                else:
                    new_population[i] = population[i]

            population = new_population.copy()

        return best_fitness, best_individual
