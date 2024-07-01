import numpy as np


class QuantumInformedEvolutionStrategy:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = np.full(self.dim, -5.0)
        self.ub = np.full(self.dim, 5.0)

    def __call__(self, func):
        population_size = 300
        elite_size = 50
        evaluations = 0
        mutation_scale = 0.2  # Increased initial mutation scale for better exploration
        recombination_prob = 0.7

        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations += population_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        while evaluations < self.budget:
            # Quantum-inspired solution space exploration
            quantum_population = np.random.uniform(self.lb, self.ub, (int(population_size * 0.2), self.dim))
            quantum_fitness = np.array([func(ind) for ind in quantum_population])
            evaluations += len(quantum_population)

            combined_population = np.vstack((population, quantum_population))
            combined_fitness = np.hstack((fitness, quantum_fitness))

            # Elite selection using tournament selection
            elite_indices = np.argsort(combined_fitness)[:elite_size]
            elite_individuals = combined_population[elite_indices]
            elite_fitness = combined_fitness[elite_indices]

            # Generate new candidates using differential evolution strategy
            new_population = []
            for _ in range(population_size - elite_size):
                indices = np.random.choice(elite_size, 3, replace=False)
                x1, x2, x3 = elite_individuals[indices]
                mutant = x1 + mutation_scale * (x2 - x3)
                mutant = np.clip(mutant, self.lb, self.ub)
                if np.random.rand() < recombination_prob:
                    cross_points = np.random.rand(self.dim) < 0.5
                    child = np.where(cross_points, mutant, x1)
                else:
                    child = mutant

                child_fitness = func(child)
                evaluations += 1

                if child_fitness < self.f_opt:
                    self.f_opt = child_fitness
                    self.x_opt = child

                new_population.append(child)

            # Update population and fitness
            population = np.vstack((elite_individuals, new_population))
            fitness = np.array([func(ind) for ind in population])
            evaluations += len(new_population)

            # Adaptive mutation scale update
            mutation_scale *= 0.98  # Slow decay to maintain explorative capabilities longer

        return self.f_opt, self.x_opt
