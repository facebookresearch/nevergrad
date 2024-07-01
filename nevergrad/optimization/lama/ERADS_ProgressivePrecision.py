import numpy as np


class ERADS_ProgressivePrecision:
    def __init__(
        self,
        budget,
        population_size=50,
        F_start=0.5,
        F_peak=0.85,
        CR=0.9,
        memory_factor=0.2,
        adaptive_CR=False,
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_start = F_start  # Initial scaling factor for mutation
        self.F_peak = F_peak  # Mid-point peak scaling factor, for aggressive exploration
        self.CR = CR  # Initial Crossover probability
        self.dimension = 5
        self.bounds = (-5.0, 5.0)  # Search space bounds
        self.memory_factor = memory_factor  # Memory factor to guide mutation based on past successful steps
        self.adaptive_CR = adaptive_CR  # Option to adapt crossover probability based on success rate

    def __call__(self, func):
        # Initialize population uniformly within the bounds
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        best_index = np.argmin(fitness)
        self.f_opt = fitness[best_index]
        self.x_opt = population[best_index]
        evaluations = self.population_size
        memory = np.zeros(self.dimension)  # Initialize memory to store successful mutation directions
        successful_CR = []

        while evaluations < self.budget:
            # Non-linear adaptation of the scaling factor
            t = evaluations / self.budget
            if t < 0.5:
                F_current = self.F_start + (self.F_peak - self.F_start) * np.sin(
                    np.pi * t
                )  # Sinusoidal increase to peak
            else:
                F_current = self.F_peak - (self.F_peak - self.F_start) * np.sin(
                    np.pi * (t - 0.5)
                )  # Symmetric decrease

            # Adaptive CR based on past successes
            if self.adaptive_CR and successful_CR:
                self.CR = np.clip(np.mean(successful_CR), 0.1, 0.9)

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
                    successful_CR.append(self.CR)
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        best_index = i
                    memory = (1 - self.memory_factor) * memory + self.memory_factor * F_current * (
                        mutant - population[i]
                    )

                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt
