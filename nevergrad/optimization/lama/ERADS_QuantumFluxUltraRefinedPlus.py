import numpy as np


class ERADS_QuantumFluxUltraRefinedPlus:
    def __init__(
        self,
        budget,
        population_size=100,
        F_init=0.5,
        F_end=0.9,
        CR=0.9,
        memory_factor=0.2,
        adaptation_rate=0.05,
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_init = F_init  # Initial scaling factor for mutation
        self.F_end = F_end  # Final scaling factor for mutation, adjusting over time
        self.CR = CR  # Crossover probability
        self.dimension = 5
        self.bounds = (-5.0, 5.0)  # Search space bounds
        self.memory_factor = memory_factor  # Memory factor to guide mutation based on past successful steps
        self.adaptation_rate = adaptation_rate  # Rate at which F and CR adapt based on the success rate

    def __call__(self, func):
        # Initialize population uniformly within the bounds
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        best_index = np.argmin(fitness)
        self.f_opt = fitness[best_index]
        self.x_opt = population[best_index]
        evaluations = self.population_size
        memory = np.zeros(self.dimension)  # Initialize memory to store successful mutation directions

        # Adaptation of F and CR based on the success of mutations and crossovers
        successful_mutation_count = 0

        while evaluations < self.budget:
            F_current = self.F_init + (self.F_end - self.F_init) * (evaluations / self.budget)

            for i in range(self.population_size):
                indices = np.random.choice(
                    [j for j in range(self.population_size) if j != i], 3, replace=False
                )
                x1, x2, x3 = population[indices]
                best = population[best_index]

                mutant = x1 + F_current * (best - x1 + x2 - x3 + self.memory_factor * memory)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                trial = np.where(np.random.rand(self.dimension) < self.CR, mutant, population[i])
                f_trial = func(trial)
                evaluations += 1

                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    successful_mutation_count += 1
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        best_index = i

                    memory = (1 - self.memory_factor) * memory + self.memory_factor * F_current * (
                        mutant - population[i]
                    )

                if evaluations >= self.budget:
                    break

            # Dynamically adjust mutation and crossover strategy based on success rate
            success_rate = successful_mutation_count / evaluations
            F_current = F_current + self.adaptation_rate * (success_rate - 0.5) * (self.F_end - self.F_init)
            self.CR = self.CR + self.adaptation_rate * (success_rate - 0.5)

        return self.f_opt, self.x_opt
