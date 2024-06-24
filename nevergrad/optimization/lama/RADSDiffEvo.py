import numpy as np


class RADSDiffEvo:
    def __init__(self, budget, population_size=100, F_base=0.5, CR_base=0.8, adaptive=True):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0
        self.F_base = F_base  # Base mutation factor
        self.CR_base = CR_base  # Base crossover probability
        self.adaptive = adaptive  # Toggle adaptive behavior

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

                # Mutation strategy selector that considers progress
                if self.adaptive:
                    progress = num_evals / self.budget
                    F = self.F_base + 0.3 * np.random.randn() * (
                        1 - progress
                    )  # Decreasing F variation over time
                    CR = self.CR_base + 0.1 * np.random.randn() * (
                        1 - progress
                    )  # Decreasing CR variation over time
                    strategy_type = "best1bin" if np.random.rand() < 0.5 * (1 + progress) else "rand1bin"
                else:
                    F = self.F_base
                    CR = self.CR_base
                    strategy_type = "best1bin" if num_evals % 2 == 0 else "rand1bin"

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

            population = new_population

        return best_fitness, best_individual
