import numpy as np


class OptimalDynamicPrecisionOptimizerV14:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5 as per the problem description
        self.lb = -5.0  # Lower bound as per the problem description
        self.ub = 5.0  # Upper bound as per the problem description

    def __call__(self, func):
        # Enhanced temperature parameters for deeper and more nuanced exploration
        T = 1.2  # Higher initial temperature to promote extensive initial search
        T_min = 0.0003  # Lower minimum temperature for fine-grained exploration at the end
        alpha = 0.93  # Slower cooling rate to ensure a gradual transition and more evaluations

        # Mutation and crossover factors fine-tuned for robust evolutionary dynamics
        F = 0.77  # Slightly increased mutation factor to induce robust exploratory mutations
        CR = 0.89  # Slightly increased crossover probability to promote diversity

        population_size = 85  # Tweaked population size for better coverage of the search space
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Implementing an advanced sigmoid function for mutation adaptation
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]

                # Mutation factor dynamically adapts with a sigmoid function for refined control
                dynamic_F = (
                    F * np.exp(-0.06 * T) * (0.6 + 0.4 * np.tanh(4 * (evaluation_count / self.budget - 0.5)))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Refined acceptance criteria incorporate a more sensitive temperature function
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.05 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Enhanced adaptive cooling strategy with sinusoidal modulation
            adaptive_cooling = alpha - 0.009 * np.cos(3 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
