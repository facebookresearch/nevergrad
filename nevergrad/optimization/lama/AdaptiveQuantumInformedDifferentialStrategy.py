import numpy as np


class AdaptiveQuantumInformedDifferentialStrategy:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = np.full(self.dim, -5.0)
        self.ub = np.full(self.dim, 5.0)

    def __call__(self, func):
        population_size = 150
        elite_size = 15
        evaluations = 0
        mutation_scale = 0.5  # Start with a higher mutation scale
        recombination_prob = 0.9  # Increase recombination probability for robust exploration
        quantum_factor = 0.1  # Reduced quantum factor to balance exploration

        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations += population_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        while evaluations < self.budget:
            # Quantum-inspired solution space exploration
            num_quantum_individuals = int(population_size * quantum_factor)
            quantum_population = np.random.uniform(self.lb, self.ub, (num_quantum_individuals, self.dim))
            quantum_fitness = np.array([func(ind) for ind in quantum_population])
            evaluations += num_quantum_individuals

            combined_population = np.vstack((population, quantum_population))
            combined_fitness = np.hstack((fitness, quantum_fitness))

            # Select the top-performing individuals as elite
            elite_indices = np.argsort(combined_fitness)[:elite_size]
            elite_individuals = combined_population[elite_indices]
            elite_fitness = combined_fitness[elite_indices]

            # Differential evolution mutation and recombination
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

            # Adapt mutation scale and quantum factor based on performance
            mutation_scale *= 0.95  # Gradual reduction of mutation scale
            if evaluations % 1000 == 0 and quantum_factor < 0.3:
                quantum_factor += 0.02  # Incrementally increase quantum factor

        return self.f_opt, self.x_opt
