import numpy as np


class QuantumEnhancedGradientClimber:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = np.full(self.dim, -5.0)
        self.ub = np.full(self.dim, 5.0)

    def __call__(self, func):
        population_size = 300  # Adjusted population size
        elite_size = 30  # Adjusted elite size for better convergence
        evaluations = 0
        mutation_factor = 0.7  # Lower initial mutation factor
        crossover_probability = 0.8  # Slightly reduced crossover probability
        quantum_probability = 0.15  # Reduced initial quantum probability
        learning_rate = 0.02  # Higher initial learning rate to accelerate convergence

        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations += population_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        previous_best = np.inf

        while evaluations < self.budget:
            if abs(previous_best - self.f_opt) < 1e-6:
                mutation_factor *= 0.98  # Reduced adaptation rate for mutation factor
                learning_rate *= 0.98  # Reduced adaptation rate for learning rate
            else:
                mutation_factor *= 1.02  # Moderated increment for escaping local minima
                learning_rate *= 1.02  # Moderated increment for gradient steps
            previous_best = self.f_opt

            # Quantum exploration step
            for _ in range(int(quantum_probability * population_size)):
                quantum_individual = np.random.uniform(self.lb, self.ub, self.dim)
                quantum_fitness = func(quantum_individual)
                evaluations += 1

                if quantum_fitness < self.f_opt:
                    self.f_opt = quantum_fitness
                    self.x_opt = quantum_individual

            # Gradient-based refinement for elites
            elite_indices = np.argsort(fitness)[:elite_size]
            for idx in elite_indices:
                gradient = np.random.normal(0, 1, self.dim)
                population[idx] += learning_rate * gradient
                population[idx] = np.clip(population[idx], self.lb, self.ub)
                new_fitness = func(population[idx])
                evaluations += 1

                if new_fitness < fitness[idx]:
                    fitness[idx] = new_fitness
                    if new_fitness < self.f_opt:
                        self.f_opt = new_fitness
                        self.x_opt = population[idx]

            # Crossover and mutation for diversity
            new_population = []
            for i in range(population_size):
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + mutation_factor * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < crossover_probability
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    new_population.append(trial)
                    fitness[i] = trial_fitness
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial

                else:
                    new_population.append(population[i])

            population = np.array(new_population)
            quantum_probability *= 1.05  # Gradual increase in quantum probability to maintain diversity

        return self.f_opt, self.x_opt
