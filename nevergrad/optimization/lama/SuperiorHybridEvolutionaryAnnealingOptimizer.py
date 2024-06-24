import numpy as np


class SuperiorHybridEvolutionaryAnnealingOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5 as per the problem description
        self.lb = -5.0  # Lower bound as per the problem description
        self.ub = 5.0  # Upper bound as per the problem description

    def __call__(self, func):
        # Initial temperature for simulated annealing, starting slightly higher for broader exploration
        T = 2.0
        T_min = 0.001  # Lower minimum temperature for finer control at late stages
        alpha = 0.90  # Cooling rate, tuned for optimal exploration-exploitation balance

        # Differential evolution parameters adjusted for robustness and efficiency
        F = 0.75  # Mutation factor adjusted for optimal mutation balance
        CR = 0.88  # Crossover probability to ensure effective trait mixing

        # Increase population size for better initial coverage and diversity
        population_size = 60
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]

                # Introduce a dynamic mutation factor that decreases as temperature drops
                dynamic_F = F * (1 - 0.05 * np.log10(1 + (1 / T)))
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Simulated annealing acceptance criterion, adjusted for a sharper probability curve
                if delta_fitness < 0 or np.random.rand() < np.exp(-delta_fitness / (T**1.5)):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Cooling schedule that dynamically adjusts based on search progress
            adaptive_cooling = alpha - 0.01 * (evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
