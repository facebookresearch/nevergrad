import numpy as np


class QuantumDynamicGradientClimberV3:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = np.full(self.dim, -5.0)
        self.ub = np.full(self.dim, 5.0)

    def __call__(self, func):
        population_size = 150  # Adjusted for a more efficient search
        elite_size = 15  # Proportionally reduced to match the new population size
        evaluations = 0
        mutation_factor = 0.4  # Lower mutation factor for focused local search
        crossover_probability = 0.6  # More conservative crossover to maintain solution stability
        quantum_probability = 0.08  # Lower initial quantum probability for focused exploration
        learning_rate = 0.005  # Further reduced learning rate for finer gradient adjustments

        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations += population_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        previous_best = np.inf

        while evaluations < self.budget:
            # Adaptive mechanism based on performance
            if abs(previous_best - self.f_opt) < 1e-7:
                mutation_factor *= 0.9
                learning_rate *= 0.9
            else:
                mutation_factor *= 1.1
                learning_rate *= 1.1
            previous_best = self.f_opt

            # Quantum exploration step with adaptive probability adjustment
            for _ in range(int(quantum_probability * population_size)):
                quantum_individual = np.random.uniform(self.lb, self.ub, self.dim)
                quantum_fitness = func(quantum_individual)
                evaluations += 1

                if quantum_fitness < self.f_opt:
                    self.f_opt = quantum_fitness
                    self.x_opt = quantum_individual

            # Elite gradient refinement with more conservative updates
            elite_indices = np.argsort(fitness)[:elite_size]
            for idx in elite_indices:
                gradient = np.random.normal(0, 1, self.dim) * learning_rate
                new_position = np.clip(population[idx] + gradient, self.lb, self.ub)
                new_fitness = func(new_position)
                evaluations += 1

                if new_fitness < fitness[idx]:
                    population[idx] = new_position
                    fitness[idx] = new_fitness
                    if new_fitness < self.f_opt:
                        self.f_opt = new_fitness
                        self.x_opt = new_position

            # Differential evolution adjustments for robust convergence
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
            quantum_probability *= 1.01  # Gradual and controlled increase in quantum probability

        return self.f_opt, self.x_opt
