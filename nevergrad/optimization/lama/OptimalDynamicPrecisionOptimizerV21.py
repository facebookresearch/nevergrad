import numpy as np


class OptimalDynamicPrecisionOptimizerV21:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality
        self.lb = -5.0  # Lower bound of the search space
        self.ub = 5.0  # Upper bound of the search space

    def __call__(self, func):
        # Further refined temperature and cooling parameters
        T = 1.2  # Increased starting temperature for broader initial exploration
        T_min = 0.0004  # Fine-tuned minimum temperature for prolonged fine-tuning phase
        alpha = 0.93  # Adjusted cooling rate allowing longer search duration and slower convergence

        # Mutation and crossover parameters optimized
        F = 0.78  # Mutation factor slightly increased for stronger explorative moves
        CR = 0.88  # Crossover probability adjusted for optimal diversity

        population_size = 85  # Fine-tuning the population size for better performance balance
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Advanced mutation strategy with dynamic adaptation based on a sigmoid function
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]

                # Enhanced dynamic mutation rate, adjusting more smoothly
                sigmoid_adjustment = 0.65 + 0.35 * np.tanh(5 * (evaluation_count / self.budget - 0.5))
                dynamic_F = F * np.exp(-0.06 * T) * sigmoid_adjustment
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # More sensitive acceptance criteria with an improved temperature function
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.07 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Enhanced adaptive cooling strategy with a sinusoidal modulation for finer control
            adaptive_cooling = alpha - 0.009 * np.sin(3 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
