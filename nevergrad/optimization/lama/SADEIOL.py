import numpy as np


class SADEIOL:
    def __init__(self, budget, population_size=50, F_init=0.5, CR_init=0.9):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F_init = F_init
        self.CR_init = CR_init

    def opposition_based_learning(self, population):
        return self.lower_bound + self.upper_bound - population

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dimension)
        )
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        success_history = []

        # Strategic looping over the budget
        while evaluations < self.budget:
            F = np.clip(np.random.normal(self.F_init, 0.1), 0.1, 1)
            CR = np.clip(np.random.normal(self.CR_init, 0.1), 0.1, 1)

            for i in range(self.population_size):
                mutation_strategy = np.random.choice(["rand", "best", "current-to-best"])
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)

                if mutation_strategy == "rand":
                    mutant = population[a] + F * (population[b] - population[c])
                elif mutation_strategy == "best":
                    mutant = population[best_idx] + F * (population[a] - population[b])
                elif mutation_strategy == "current-to-best":
                    mutant = (
                        population[i]
                        + F * (population[best_idx] - population[i])
                        + F * (population[a] - population[b])
                    )

                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dimension) < CR
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    success_history.append(1)
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness
                else:
                    success_history.append(0)

                if len(success_history) > 50:  # Update rates based on recent history
                    success_rate = np.mean(success_history[-50:])
                    self.F_init = F * success_rate
                    self.CR_init = CR * success_rate

            if (
                evaluations % (self.population_size * 10) == 0 and np.std(fitness) < 1e-5
            ):  # Stagnation detection
                population = self.opposition_based_learning(population)
                fitness = np.array([func(ind) for ind in population])
                evaluations += self.population_size

        return best_fitness, best_solution
