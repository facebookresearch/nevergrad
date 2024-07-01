import numpy as np


class UltraOptimizedDynamicPrecisionOptimizerV19:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality remains fixed at 5
        self.lb = -5.0  # Lower bound of the search space
        self.ub = 5.0  # Upper bound of the search space

    def __call__(self, func):
        # Initialize temperature and cooling parameters with refined settings
        T = 1.2  # Increased starting temperature for more aggressive global exploration initially
        T_min = 0.0003  # Lower minimum temperature to allow very detailed exploration at the end
        alpha = 0.93  # Slower cooling rate to extend the search process over a longer time frame

        # Refined mutation and crossover parameters for improved performance
        F = 0.78  # Mutation factor adjusted to enhance the balance of exploration and exploitation
        CR = 0.88  # Crossover probability finely tuned for better trait integration

        population_size = 85  # Adjusted population size to optimize computational resources and diversity
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Implementing an enhanced dynamic mutation strategy with adaptive sigmoid function
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]

                # Dynamic mutation rate adapting with a more advanced sigmoid model
                dynamic_F = (
                    F
                    * np.exp(-0.06 * T)
                    * (0.75 + 0.25 * np.tanh(5 * (evaluation_count / self.budget - 0.5)))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Advanced acceptance criteria with temperature-sensitive decision making
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.07 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Advanced cooling strategy integrating a periodic modulation for nuanced temperature control
            adaptive_cooling = alpha - 0.009 * np.sin(2 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
