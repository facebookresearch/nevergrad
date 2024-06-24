import numpy as np


class DynamicCohortAdaptiveEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0

        self.initial_population_size = 50
        self.cohort_size = 10  # Size of subpopulations
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.local_search_chance = 0.2  # Probability to perform local search

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.initial_population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]

        evaluations = self.initial_population_size

        while evaluations < self.budget:
            # Split population into cohorts
            cohorts = [
                population[i : i + self.cohort_size] for i in range(0, len(population), self.cohort_size)
            ]
            new_population = []

            for cohort in cohorts:
                if len(cohort) < self.cohort_size:
                    continue  # Skip incomplete cohorts

                for i in range(len(cohort)):
                    # Mutation step
                    idxs = [idx for idx in range(len(cohort)) if idx != i]
                    a, b, c = cohort[np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + self.F * (b - c), self.lb, self.ub)

                    # Crossover step
                    crossover = np.random.rand(self.dim) < self.CR
                    trial = np.where(crossover, mutant, cohort[i])

                    # Local search based on chance
                    if np.random.rand() < self.local_search_chance:
                        trial = self.local_search(trial, func)

                    # Selection step
                    f_trial = func(trial)
                    evaluations += 1
                    if f_trial < fitness[i]:
                        new_population.append(trial)
                        if f_trial < self.f_opt:
                            self.f_opt = f_trial
                            self.x_opt = trial
                    else:
                        new_population.append(cohort[i])

                    if evaluations >= self.budget:
                        break

            population = np.array(new_population)
            fitness = np.array([func(ind) for ind in population])

            # Adaptive control of parameters
            self.adaptive_F_CR(evaluations)

        return self.f_opt, self.x_opt

    def local_search(self, x, func):
        best_x = x.copy()
        best_f = func(x)

        for _ in range(10):  # Local search iterations
            for i in range(self.dim):
                x_new = best_x.copy()
                step_size = np.random.uniform(-0.1, 0.1)
                x_new[i] = np.clip(best_x[i] + step_size, self.lb, self.ub)
                f_new = func(x_new)

                if f_new < best_f:
                    best_x = x_new
                    best_f = f_new

        return best_x

    def adaptive_F_CR(self, evaluations):
        # Adaptive parameters adjustment
        if evaluations % 100 == 0:
            self.F = np.random.uniform(0.4, 0.9)
            self.CR = np.random.uniform(0.1, 0.9)
            self.local_search_chance = np.random.uniform(0.1, 0.3)
