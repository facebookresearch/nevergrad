import numpy as np


class AdaptiveGradientEvolution:
    def __init__(self, budget):
        self.budget = budget
        self.dim = 5
        self.bounds = [-5.0, 5.0]
        self.population_size = 10
        self.base_lr = 0.1
        self.epsilon = 1e-8
        self.crossover_rate = 0.7
        self.mutation_rate = 0.1

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

        def random_vector():
            return np.random.uniform(self.bounds[0], self.bounds[1], self.dim)

        def gradient_estimate(x, h=1e-5):
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x1 = np.copy(x)
                x2 = np.copy(x)
                x1[i] += h
                x2[i] -= h
                grad[i] = (func(x1) - func(x2)) / (2 * h)
            return grad

        def adaptive_learning_rate(base_lr, iteration, success_rate):
            return base_lr / (1 + iteration * success_rate)

        # Initialize population
        population = [random_vector() for _ in range(self.population_size)]
        fitness = [func(ind) for ind in population]

        for ind, fit in zip(population, fitness):
            if fit < self.f_opt:
                self.f_opt = fit
                self.x_opt = ind

        for i in range(1, self.budget):
            success_count = 0
            # Selection: Choose two parents based on fitness
            parents_idx = np.random.choice(range(self.population_size), size=2, replace=False)
            parent1, parent2 = population[parents_idx[0]], population[parents_idx[1]]

            # Crossover
            if np.random.rand() < self.crossover_rate:
                cross_point = np.random.randint(1, self.dim - 1)
                child = np.concatenate((parent1[:cross_point], parent2[cross_point:]))
            else:
                child = parent1.copy()

            # Mutation
            if np.random.rand() < self.mutation_rate:
                mutation_idx = np.random.randint(self.dim)
                child[mutation_idx] = np.random.uniform(self.bounds[0], self.bounds[1])

            # Gradient-based exploitation
            grad = gradient_estimate(child)
            success_rate = success_count / max(1, i)  # Avoid division by zero
            adapt_lr = adaptive_learning_rate(self.base_lr, i, success_rate)
            perturbation = np.random.randn(self.dim) * adapt_lr
            new_x = child - adapt_lr * grad + perturbation

            new_x = np.clip(new_x, self.bounds[0], self.bounds[1])
            new_f = func(new_x)

            if new_f < self.f_opt:
                self.f_opt = new_f
                self.x_opt = new_x
                success_count += 1

            # Replace the worse parent with the new child
            worse_parent_idx = (
                parents_idx[0] if fitness[parents_idx[0]] > fitness[parents_idx[1]] else parents_idx[1]
            )
            population[worse_parent_idx] = new_x
            fitness[worse_parent_idx] = new_f

        return self.f_opt, self.x_opt


# Example of usage:
# optimizer = AdaptiveGradientEvolution(budget=1000)
# best_value, best_solution = optimizer(some_black_box_function)
