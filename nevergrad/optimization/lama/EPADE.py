import numpy as np


class EPADE:
    def __init__(self, budget, population_size=50, F_base=0.8, CR_base=0.9):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Enhanced base differential weight
        self.CR_base = CR_base  # Enhanced base crossover probability
        self.dimension = 5
        self.bounds = (-5.0, 5.0)  # Search space bounds

    def __call__(self, func):
        # Initialize population and fitness
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        evaluations = self.population_size

        # Adaptive parameters initialization
        F = np.full(self.population_size, self.F_base)
        CR = np.full(self.population_size, self.CR_base)

        # Introduction of an intensive search in early phase
        intensive_search_phase = True

        while evaluations < self.budget:
            # Gradual transition from intensive to conservative strategy based on the budget usage
            if evaluations > self.budget * 0.5:
                intensive_search_phase = False

            for i in range(self.population_size):
                # Adaptive mutation factor based on individual performance
                if fitness[i] < np.median(fitness):
                    F[i] = min(F[i] * 1.2, 1) if intensive_search_phase else min(F[i] * 1.1, 1)
                else:
                    F[i] = max(F[i] * 0.8, 0.1) if intensive_search_phase else max(F[i] * 0.9, 0.1)

                # Mutation and crossover using a mixed strategy
                idxs = np.random.choice([j for j in range(self.population_size) if j != i], 3, replace=False)
                best_idx = np.argmin(fitness)
                mutant = population[i] + F[i] * (
                    population[best_idx] - population[i] + population[idxs[0]] - population[idxs[1]]
                )
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                cross_points = np.random.rand(self.dimension) < CR[i]
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dimension)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Trial solution evaluation
                f_trial = func(trial)
                evaluations += 1

                # Selection and adaptive CR update based on success
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    CR[i] = min(CR[i] * 1.1, 1)
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    CR[i] = max(CR[i] * 0.85, 0.1)

                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt
