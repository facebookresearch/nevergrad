import numpy as np


class ERADS_UltraDynamicMaxUltraPlus:
    def __init__(self, budget, population_size=45, F_init=0.5, F_end=0.9, CR=0.88, memory_factor=0.25):
        self.budget = budget
        self.population_size = population_size
        self.F_init = F_init  # Initial scaling factor for mutations
        self.F_end = (
            F_end  # Final scaling factor for mutations, increasing over time for aggressive late exploration
        )
        self.CR = CR  # Crossover probability, slightly reduced to maintain good solutions
        self.dimension = 5
        self.bounds = (-5.0, 5.0)  # Search space bounds
        self.memory_factor = memory_factor  # Memory factor to enhance mutation strategy based on past success

    def __call__(self, func):
        # Initialize population uniformly within the bounds
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        best_index = np.argmin(fitness)
        self.f_opt = fitness[best_index]
        self.x_opt = population[best_index]
        evaluations = self.population_size
        memory = np.zeros(self.dimension)  # Initialize memory for successful mutation directions

        while evaluations < self.budget:
            # Adapt mutation factor linearly over the course of the optimization
            F_current = self.F_init + (self.F_end - self.F_init) * (evaluations / self.budget)

            for i in range(self.population_size):
                # Select three different random indices, excluding the current index i
                indices = np.random.choice(
                    [j for j in range(self.population_size) if j != i], 3, replace=False
                )
                x1, x2, x3 = population[indices]
                best = population[best_index]

                # Generate mutant vector using current population, best solution, and memory
                mutant = x1 + F_current * (best - x1 + x2 - x3 + self.memory_factor * memory)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])  # Ensure mutant is within bounds

                # Perform crossover to create the trial solution
                trial = np.where(np.random.rand(self.dimension) < self.CR, mutant, population[i])
                f_trial = func(trial)
                evaluations += 1

                # Update if the trial solution is better
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        best_index = i  # Update best index

                    # Update memory with successful mutation scaled by mutation factor
                    memory = (1 - self.memory_factor) * memory + self.memory_factor * F_current * (
                        mutant - population[i]
                    )

                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt
