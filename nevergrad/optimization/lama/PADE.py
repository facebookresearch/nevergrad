import numpy as np


class PADE:
    def __init__(self, budget, population_size=50, F_base=0.5, CR_base=0.9):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Base differential weight
        self.CR_base = CR_base  # Base crossover probability
        self.dimension = 5
        self.bounds = (-5.0, 5.0)  # Search space bounds

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])

        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        evaluations = self.population_size

        # Initialize adaptive parameters
        F = np.full(self.population_size, self.F_base)
        CR = np.full(self.population_size, self.CR_base)

        # Early and late phase indicators
        early_phase = True

        while evaluations < self.budget:
            # Determine phase transition based on budget usage
            if evaluations > self.budget * 0.5:
                early_phase = False

            for i in range(self.population_size):
                # Adaptive mutation strategy depending on phase and fitness
                if fitness[i] < np.median(fitness):
                    F[i] *= 1.1 if early_phase else 1.05
                    F[i] = min(F[i], 1)
                else:
                    F[i] *= 0.9 if early_phase else 0.95
                    F[i] = max(F[i], 0.1)

                # Mutation and crossover using "best" and "rand" strategy combination
                idxs = np.random.choice(
                    [idx for idx in range(self.population_size) if idx != i], 2, replace=False
                )
                best_idx = np.argmin(fitness)
                mutant = population[i] + F[i] * (
                    population[best_idx] - population[i] + population[idxs[0]] - population[idxs[1]]
                )
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                cross_points = np.random.rand(self.dimension) < CR[i]
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dimension)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate trial solution
                f_trial = func(trial)
                evaluations += 1

                # Selection step
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial

                    # Update optimal found solution
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                # Adaptive CR adjustment
                CR[i] = CR[i] * 1.1 if f_trial < fitness[i] else CR[i] * 0.9
                CR[i] = min(max(CR[i], 0.1), 1)

                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt
