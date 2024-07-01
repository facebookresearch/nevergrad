import numpy as np


class HyperOptimizedDynamicPrecisionOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5 as per the problem description
        self.lb = -5.0  # Lower bound
        self.ub = 5.0  # Upper bound

    def __call__(self, func):
        # Initialize advanced temperature and cooling parameters
        T = 1.2  # Initial temperature set higher for more aggressive exploration early on
        T_min = 0.0005  # Lower minimum temperature for detailed exploration late in the search
        alpha = 0.92  # Slower cooling rate to maintain exploration capabilities longer

        population_size = 80  # Adjusted population size for better coverage of the search space
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Enhancing mutation strategy and adaptive acceptance criteria
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Temperature and evaluation count influenced dynamic mutation factor
                dynamic_F = (
                    0.8
                    * np.exp(-0.15 * T)
                    * (0.7 + 0.3 * np.cos(1.2 * np.pi * evaluation_count / self.budget))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                CR = 0.85 + 0.1 * np.sin(
                    2 * np.pi * evaluation_count / self.budget
                )  # Dynamic crossover probability
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Adapting the acceptance criterion to be more aggressive based on fitness improvements
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.07 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Fine-tuning the cooling strategy based on the search process dynamics
            adaptive_cooling = alpha - 0.015 * np.cos(1.5 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
