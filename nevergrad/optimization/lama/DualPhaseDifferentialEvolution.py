import numpy as np


class DualPhaseDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.dim = 5

    def __call__(self, func):
        # Initialize parameters
        population_size = 50
        initial_F = 0.8  # Initial Differential weight
        initial_CR = 0.9  # Initial Crossover probability
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        best_position = None
        best_value = np.Inf

        eval_count = 0
        phase_switch_threshold = self.budget // 2

        def quantum_position_update(position, best_position):
            return position + np.random.uniform(-1, 1, position.shape) * (best_position - position) / 2

        def adapt_parameters(eval_count, budget):
            # Adaptive strategy for F and CR
            adaptive_F = initial_F * (1 - eval_count / budget)
            adaptive_CR = initial_CR * np.cos(np.pi * eval_count / (2 * budget))
            return adaptive_F, adaptive_CR

        while eval_count < self.budget:
            new_population = np.copy(population)
            for i in range(population_size):
                if eval_count >= self.budget:
                    break
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                F, CR = adapt_parameters(eval_count, self.budget)
                mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
                trial = np.copy(population[i])
                for d in range(self.dim):
                    if np.random.rand() < CR:
                        trial[d] = mutant[d]

                if eval_count < phase_switch_threshold:
                    candidate = trial  # Exploration phase, use the trial vector
                else:
                    candidate = quantum_position_update(
                        trial, best_position if best_position is not None else trial
                    )

                candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                candidate_value = func(candidate)
                eval_count += 1

                if candidate_value < best_value:
                    best_value = candidate_value
                    best_position = candidate
                    new_population[i] = candidate
                else:
                    new_population[i] = trial

            population = new_population

        return best_value, best_position


# Example usage:
# func = SomeBlackBoxFunction()  # The black box function to be optimized
# optimizer = DualPhaseDifferentialEvolution(budget=10000)
# best_value, best_position = optimizer(func)
