import numpy as np


class EADEPC:
    def __init__(
        self,
        budget,
        dimension=5,
        lower_bound=-5.0,
        upper_bound=5.0,
        population_size=40,
        F_init=0.5,
        CR_init=0.9,
    ):
        self.budget = budget
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population_size = population_size
        self.F = F_init
        self.CR = CR_init

    def __call__(self, func):
        # Initialize population uniformly
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dimension)
        )
        fitness = np.array([func(individual) for individual in population])
        evaluations = self.population_size

        best_index = np.argmin(fitness)
        best_individual = population[best_index]
        best_fitness = fitness[best_index]

        # Adaptive control parameters
        success_counter = 0
        adapt_frequency = max(1, int(0.1 * self.budget))

        while evaluations < self.budget:
            F_adapted = np.clip(
                np.random.normal(self.F, 0.1), 0.1, 1.0
            )  # Normal distribution around F with clipping
            CR_adapted = np.clip(
                np.random.normal(self.CR, 0.1), 0.1, 1.0
            )  # Normal distribution around CR with clipping

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Mutation using "rand/1/bin" strategy
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = population[a] + F_adapted * (population[b] - population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                trial = np.array(
                    [
                        mutant[j] if np.random.rand() < CR_adapted else population[i][j]
                        for j in range(self.dimension)
                    ]
                )

                # Selection
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    success_counter += 1

                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial

            # Adjust F and CR based on performance after a certain number of evaluations
            if evaluations % adapt_frequency == 0:
                success_rate = success_counter / (self.population_size * adapt_frequency)
                self.F = self.F * (0.9 if success_rate < 0.2 else 1.1)
                self.CR = self.CR * (0.9 if success_rate > 0.2 else 1.1)
                self.F = max(0.1, min(self.F, 0.9))
                self.CR = max(0.1, min(self.CR, 0.9))
                success_counter = 0  # Reset success counter

        return best_fitness, best_individual
