import numpy as np


class HybridGradientEvolution:
    def __init__(self, budget):
        self.budget = budget
        self.dim = 5
        self.bounds = [-5.0, 5.0]
        self.population_size = 10
        self.learning_rate = 0.1
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

        # Initialize population
        population = [random_vector() for _ in range(self.population_size)]
        fitness = [func(ind) for ind in population]

        for ind, fit in zip(population, fitness):
            if fit < self.f_opt:
                self.f_opt = fit
                self.x_opt = ind

        for i in range(1, self.budget):
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
            adapt_lr = self.learning_rate / (np.sqrt(i) + self.epsilon)
            perturbation = np.random.randn(self.dim) * adapt_lr
            new_x = child - adapt_lr * grad + perturbation

            new_x = np.clip(new_x, self.bounds[0], self.bounds[1])
            new_f = func(new_x)

            if new_f < self.f_opt:
                self.f_opt = new_f
                self.x_opt = new_x

            # Replace the worse parent with the new child
            worse_parent_idx = (
                parents_idx[0] if fitness[parents_idx[0]] > fitness[parents_idx[1]] else parents_idx[1]
            )
            population[worse_parent_idx] = new_x
            fitness[worse_parent_idx] = new_f

        return self.f_opt, self.x_opt


# Example of usage:
# optimizer = HybridGradientEvolution(budget=1000)
# best_value, best_solution = optimizer(some_black_box_function)
