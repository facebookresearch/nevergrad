import numpy as np


class HybridGradientCrossoverOptimization:
    def __init__(
        self, budget, dimension=5, population_size=30, learning_rate=0.1, crossover_rate=0.7, gradient_steps=5
    ):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.learning_rate = learning_rate
        self.crossover_rate = crossover_rate
        self.gradient_steps = gradient_steps  # Number of gradient steps after crossover/mutation

    def __call__(self, func):
        # Initialize population within bounds [-5.0, 5.0]
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dimension))
        fitness = np.array([func(individual) for individual in population])
        best_idx = np.argmin(fitness)
        f_opt = fitness[best_idx]
        x_opt = population[best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Select parents for crossover
                parents_idx = np.random.choice(self.population_size, 2, replace=False)
                parent1, parent2 = population[parents_idx[0]], population[parents_idx[1]]

                # Perform crossover
                if np.random.rand() < self.crossover_rate:
                    child = np.array([np.random.choice([p1, p2]) for p1, p2 in zip(parent1, parent2)])
                else:
                    child = parent1.copy()  # No crossover, child is a copy of parent1

                # Mutation: adding Gaussian noise
                child += np.random.normal(0, 1, self.dimension) * self.learning_rate
                child = np.clip(child, -5.0, 5.0)  # Ensure child is within bounds

                # Evaluate child
                child_fitness = func(child)
                evaluations += 1

                # Selection: Greedily replace if the child is better
                if child_fitness < fitness[i]:
                    population[i] = child
                    fitness[i] = child_fitness

                # Gradient-based refinement for a few steps
                if evaluations + self.gradient_steps <= self.budget:
                    for _ in range(self.gradient_steps):
                        grad_est = np.array(
                            [
                                (func(child + eps * np.eye(1, self.dimension, k)[0]) - child_fitness) / eps
                                for k, eps in enumerate([1e-5] * self.dimension)
                            ]
                        )
                        child -= self.learning_rate * grad_est
                        child = np.clip(child, -5.0, 5.0)
                        new_fitness = func(child)
                        evaluations += 1

                        if new_fitness < child_fitness:
                            child_fitness = new_fitness
                            population[i] = child
                            fitness[i] = child_fitness
                        else:
                            break

                # Update global optimum
                if child_fitness < f_opt:
                    f_opt = child_fitness
                    x_opt = child

                if evaluations >= self.budget:
                    break

        return f_opt, x_opt
