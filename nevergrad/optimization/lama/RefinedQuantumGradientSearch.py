import numpy as np


class RefinedQuantumGradientSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = np.full(self.dim, -5.0)
        self.ub = np.full(self.dim, 5.0)

    def __call__(self, func):
        population_size = 300  # Further increased population size for more exploration
        elite_size = 30  # Slightly increased elite size for better exploitation
        evaluations = 0
        mutation_factor = 0.75  # Adjusted mutation factor
        crossover_probability = 0.95  # Very high crossover probability to better mix genetic information
        quantum_probability = 0.15  # Increased initial quantum probability for aggressive exploration
        learning_rate = 0.008  # Slightly decreased learning rate for more stable gradient descent

        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations += population_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        previous_best = np.inf

        while evaluations < self.budget:
            if abs(previous_best - self.f_opt) < 1e-7:  # More sensitive threshold for convergence detection
                mutation_factor *= 0.9  # More controlled mutation factor decrement
                learning_rate *= 0.9  # Reduce learning rate to stabilize near minima
            else:
                mutation_factor *= 1.1  # Increment mutation factor to explore aggressively
                learning_rate *= 1.1  # Increase learning rate for faster convergence on hills
            previous_best = self.f_opt

            for _ in range(int(quantum_probability * population_size)):
                quantum_individual = np.random.uniform(self.lb, self.ub, self.dim)
                quantum_fitness = func(quantum_individual)
                evaluations += 1

                if quantum_fitness < self.f_opt:
                    self.f_opt = quantum_fitness
                    self.x_opt = quantum_individual

            elite_indices = np.argsort(fitness)[:elite_size]
            for idx in elite_indices:
                gradient = np.random.normal(0, 1, self.dim)  # Simulated gradient representation
                population[idx] += learning_rate * gradient
                population[idx] = np.clip(population[idx], self.lb, self.ub)
                new_fitness = func(population[idx])
                evaluations += 1

                if new_fitness < fitness[idx]:
                    fitness[idx] = new_fitness
                    if new_fitness < self.f_opt:
                        self.f_opt = new_fitness
                        self.x_opt = population[idx]

            new_population = []
            for i in range(population_size):
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = a + mutation_factor * (b - c)
                mutant = np.clip(mutant, self.lb, self.ub)
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
            quantum_probability = min(0.25, quantum_probability * 1.1)  # Gently increase quantum probability

        return self.f_opt, self.x_opt
