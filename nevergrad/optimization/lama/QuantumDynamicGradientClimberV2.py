import numpy as np


class QuantumDynamicGradientClimberV2:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = np.full(self.dim, -5.0)
        self.ub = np.full(self.dim, 5.0)

    def __call__(self, func):
        population_size = 200  # Reduced for focused search
        elite_size = 20  # Reduced elite size
        evaluations = 0
        mutation_factor = 0.5  # Lower mutation factor to reduce drastic changes
        crossover_probability = 0.7  # Reduced crossover probability for more stable evolution
        quantum_probability = 0.10  # Initial quantum probability
        learning_rate = 0.01  # Reduced learning rate for more precise gradient steps

        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations += population_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        previous_best = np.inf

        while evaluations < self.budget:
            # Adaptive mechanism based on stagnation detection
            if abs(previous_best - self.f_opt) < 1e-6:
                mutation_factor *= 0.95
                learning_rate *= 0.95
            else:
                mutation_factor *= 1.05
                learning_rate *= 1.05
            previous_best = self.f_opt

            # Quantum exploration step with adaptive probability increase
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
                gradient = np.random.normal(0, 1, self.dim) * learning_rate
                population[idx] += gradient
                population[idx] = np.clip(population[idx], self.lb, self.ub)
                new_fitness = func(population[idx])
                evaluations += 1

                if new_fitness < fitness[idx]:
                    fitness[idx] = new_fitness
                    if new_fitness < self.f_opt:
                        self.f_opt = new_fitness
                        self.x_opt = population[idx]

            # Differential evolution for the rest of the population
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
            quantum_probability *= 1.02  # Gradual increase in quantum probability

        return self.f_opt, self.x_opt
