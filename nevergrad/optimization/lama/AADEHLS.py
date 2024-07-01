import numpy as np


class AADEHLS:
    def __init__(self, budget, population_size=50, F_init=0.5, CR_init=0.9):
        self.budget = budget
        self.CR_init = CR_init
        self.F_init = F_init
        self.population_size = population_size
        self.dimension = 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def opposite_point(self, x):
        return self.lower_bound + self.upper_bound - x

    def __call__(self, func):
        # Initialize population with Opposition-Based Learning
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dimension)
        )
        opposite_population = self.opposite_point(population)
        combined_population = np.vstack((population, opposite_population))
        fitness = np.array([func(ind) for ind in combined_population])
        indices = np.argsort(fitness)
        population = combined_population[indices[: self.population_size]]
        fitness = fitness[indices[: self.population_size]]

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        F = self.F_init
        CR = self.CR_init
        successful_F = []
        successful_CR = []

        evaluations = self.population_size * 2
        while evaluations < self.budget:
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = population[a] + F * (population[b] - population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                cross_points = np.random.rand(self.dimension) < CR
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    successful_F.append(F)
                    successful_CR.append(CR)
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

            # Update F and CR adaptively based on successes
            if successful_F:
                F = np.mean(successful_F)
                CR = np.mean(successful_CR)

            # Enhanced hybrid local search phase
            local_best = best_solution.copy()
            for _ in range(10):
                perturbation = np.random.normal(0, 0.1, self.dimension)
                local_trial = np.clip(local_best + perturbation, self.lower_bound, self.upper_bound)
                local_fitness = func(local_trial)
                evaluations += 1

                if local_fitness < best_fitness:
                    best_solution = local_trial
                    best_fitness = local_fitness
                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
