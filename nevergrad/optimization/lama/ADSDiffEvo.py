import numpy as np


class ADSDiffEvo:
    def __init__(self, budget, population_size=100, F_base=0.6, CR_base=0.7):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0
        self.F_base = F_base  # Base mutation factor
        self.CR_base = CR_base  # Base crossover probability

    def __call__(self, func):
        # Initialize population and fitness evaluations
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

                # Adaptive strategy, alternating mutation strategies
                if num_evals % 2 == 0:
                    strategy_type = "rand1bin"
                else:
                    strategy_type = "best1bin"

                F = self.F_base + 0.1 * np.random.randn()  # Perturbed mutation factor
                CR = self.CR_base + 0.1 * np.random.randn()  # Perturbed crossover rate

                idxs = [idx for idx in range(self.population_size) if idx != i]
                chosen = np.random.choice(idxs, 3, replace=False)
                a, b, c = population[chosen]

                # Mutation strategies
                if strategy_type == "rand1bin":
                    mutant = a + F * (b - c)
                elif strategy_type == "best1bin":
                    mutant = best_individual + F * (b - c)

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
