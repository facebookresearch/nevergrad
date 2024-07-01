import numpy as np


class RefinedQuantumInformedDifferentialStrategyV2:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = np.full(self.dim, -5.0)
        self.ub = np.full(self.dim, 5.0)

    def __call__(self, func):
        population_size = 100
        elite_size = 10
        evaluations = 0
        mutation_scale = 0.8  # Increased initial mutation scale
        recombination_prob = 0.95  # Higher recombination probability
        quantum_factor = 0.05  # Initial quantum factor
        convergence_threshold = 1e-5  # Threshold to enhance convergence monitoring

        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations += population_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        previous_best = np.inf

        while evaluations < self.budget:
            # Check for convergence improvement
            if abs(previous_best - self.f_opt) < convergence_threshold:
                mutation_scale *= 0.9  # Reduce the mutation scale to fine-tune the search
            previous_best = self.f_opt

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
                child = np.where(np.random.rand(self.dim) < recombination_prob, mutant, x1)

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

            # Dynamically adapt quantum factor based on convergence
            if evaluations % 1000 == 0:
                quantum_factor = min(0.5, quantum_factor + 0.05)  # Gradually increase the quantum factor

        return self.f_opt, self.x_opt
