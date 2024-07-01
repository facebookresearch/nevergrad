import numpy as np


class RefinedHybridAdaptiveDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dimension = 5  # Dimensionality of the problem
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 200  # Further increased population for greater exploratory potential
        self.crossover_rate = 0.9  # Higher crossover rate for better gene mixing
        self.differential_weight = (
            0.75  # Fine-tuned differential weight for robust exploration-exploitation balance
        )
        self.patience = 30  # Reduced patience for more dynamic adaptation
        self.p_adaptive_mutation = 0.2  # Mutation probability starts lower
        self.adaptation_factor = 1.05  # Controlled adaptation rate for mutation probability

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dimension)
        )
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        evaluations = self.population_size
        generations_since_last_improvement = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.differential_weight * (b - c), self.lower_bound, self.upper_bound)

                # Cross-over operation
                cross_points = np.random.rand(self.dimension) < self.crossover_rate
                trial = np.where(cross_points, mutant, population[i])
                f_trial = func(trial)
                evaluations += 1

                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial

                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        generations_since_last_improvement = 0
                else:
                    generations_since_last_improvement += 1

                if evaluations >= self.budget:
                    break

            if generations_since_last_improvement > self.patience:
                # Adapt mutation probability dynamically to enhance exploration
                generations_since_last_improvement = 0
                self.p_adaptive_mutation = min(1, self.p_adaptive_mutation * self.adaptation_factor)

        return self.f_opt, self.x_opt
