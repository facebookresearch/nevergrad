import numpy as np


class DynamicPopulationAdaptiveGradientEvolution:
    def __init__(self, budget, initial_population_size=10, max_population_size=50):
        self.budget = budget
        self.dim = 5
        self.bounds = [-5.0, 5.0]
        self.initial_population_size = initial_population_size
        self.max_population_size = max_population_size
        self.base_lr = 0.1
        self.epsilon = 1e-8
        self.crossover_rate = 0.7
        self.mutation_rate = 0.1
        self.diversity_threshold = 1e-3

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

        def maintain_diversity(population, fitness):
            for i in range(len(population)):
                for j in range(i + 1, len(population)):
                    if np.linalg.norm(population[i] - population[j]) < self.diversity_threshold:
                        if fitness[i] > fitness[j]:
                            population[i] = random_vector()
                        else:
                            population[j] = random_vector()

        def dynamic_population_adjustment(population, fitness, iteration):
            if iteration % 10 == 0 and len(population) < self.max_population_size:
                population.append(random_vector())
                fitness.append(func(population[-1]))

        # Initialize population
        population = [random_vector() for _ in range(self.initial_population_size)]
        fitness = [func(ind) for ind in population]

        for ind, fit in zip(population, fitness):
            if fit < self.f_opt:
                self.f_opt = fit
                self.x_opt = ind

        iteration = 0
        success_count = 0
        while iteration < self.budget:
            # Selection: Choose two parents based on fitness
            parents_idx = np.random.choice(range(len(population)), size=2, replace=False)
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
            success_rate = success_count / max(1, iteration)  # Avoid division by zero
            adapt_lr = adaptive_learning_rate(self.base_lr, iteration, success_rate)
            perturbation = np.random.randn(self.dim) * adapt_lr
            new_x = child - adapt_lr * grad + perturbation

            new_x = np.clip(new_x, self.bounds[0], self.bounds[1])
            new_f = func(new_x)
            iteration += 1

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

            # Maintain diversity
            maintain_diversity(population, fitness)
            # Dynamically adjust population size
            dynamic_population_adjustment(population, fitness, iteration)

        return self.f_opt, self.x_opt


# Example of usage:
# optimizer = DynamicPopulationAdaptiveGradientEvolution(budget=1000)
# best_value, best_solution = optimizer(some_black_box_function)
