import numpy as np


class HybridAdaptiveDiversityMaintainingGradientEvolution:
    def __init__(self, budget, initial_population_size=20):
        self.budget = budget
        self.dim = 5
        self.bounds = [-5.0, 5.0]
        self.population_size = initial_population_size
        self.base_lr = 0.1
        self.epsilon = 1e-8
        self.crossover_rate = 0.7
        self.mutation_rate = 0.1
        self.diversity_threshold = 1e-3
        self.elite_rate = 0.2  # Proportion of elite members in selection
        self.local_search_rate = 0.3  # Probability to perform local search

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

        def elite_selection(population, fitness):
            elite_count = int(self.elite_rate * len(fitness))
            sorted_indices = np.argsort(fitness)
            elite_indices = sorted_indices[:elite_count]
            return [population[i] for i in elite_indices], [fitness[i] for i in elite_indices]

        def local_search(x):
            grad = gradient_estimate(x)
            step = -self.base_lr * grad
            new_x = x + step
            new_x = np.clip(new_x, self.bounds[0], self.bounds[1])
            return new_x

        # Initialize population
        population = [random_vector() for _ in range(self.population_size)]
        fitness = [func(ind) for ind in population]

        for ind, fit in zip(population, fitness):
            if fit < self.f_opt:
                self.f_opt = fit
                self.x_opt = ind

        for i in range(1, self.budget):
            success_count = 0

            # Elite selection
            elite_pop, elite_fit = elite_selection(population, fitness)
            elite_size = len(elite_pop)

            if np.random.rand() < self.local_search_rate:
                # Local search
                local_idx = np.random.choice(range(elite_size), size=1)[0]
                child = local_search(elite_pop[local_idx])
            else:
                # Selection: Choose two parents based on fitness
                parents_idx = np.random.choice(range(elite_size), size=2, replace=False)
                parent1, parent2 = elite_pop[parents_idx[0]], elite_pop[parents_idx[1]]

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

                # Replace the worst member of the population with the new child
                worst_idx = np.argmax(fitness)
                population[worst_idx] = new_x
                fitness[worst_idx] = new_f

            # Maintain diversity
            maintain_diversity(population, fitness)

        return self.f_opt, self.x_opt


# Example of usage:
# optimizer = HybridAdaptiveDiversityMaintainingGradientEvolution(budget=1000)
# best_value, best_solution = optimizer(some_black_box_function)
