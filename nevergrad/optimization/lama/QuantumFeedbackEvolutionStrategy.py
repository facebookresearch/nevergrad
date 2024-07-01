import numpy as np


class QuantumFeedbackEvolutionStrategy:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0 * np.ones(self.dim)
        self.ub = 5.0 * np.ones(self.dim)

    def __call__(self, func):
        population_size = 100
        elite_size = 10
        evaluations = 0
        mutation_factor = 0.5
        crossover_probability = 0.7
        quantum_probability = 0.05
        feedback_threshold = 1e-6
        feedback_gain = 0.1

        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations += population_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        previous_best = self.f_opt

        while evaluations < self.budget:
            # Evolve using differential evolution
            new_population = np.empty_like(population)
            for i in range(population_size):
                indices = [j for j in range(population_size) if j != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + mutation_factor * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < crossover_probability
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                else:
                    new_population[i] = population[i]

                if trial_fitness < self.f_opt:
                    self.f_opt = trial_fitness
                    self.x_opt = trial

            population = new_population

            # Feedback mechanism for dynamic adaptation
            if abs(previous_best - self.f_opt) < feedback_threshold:
                mutation_factor -= feedback_gain * mutation_factor
                crossover_probability += feedback_gain * (1 - crossover_probability)
            else:
                mutation_factor += feedback_gain * (1 - mutation_factor)
                crossover_probability -= feedback_gain * crossover_probability

            previous_best = self.f_opt

            # Quantum mutation for exploration
            if np.random.rand() < quantum_probability:
                quantum_individual = np.random.uniform(self.lb, self.ub, self.dim)
                quantum_fitness = func(quantum_individual)
                evaluations += 1

                if quantum_fitness < self.f_opt:
                    self.f_opt = quantum_fitness
                    self.x_opt = quantum_individual

        return self.f_opt, self.x_opt
