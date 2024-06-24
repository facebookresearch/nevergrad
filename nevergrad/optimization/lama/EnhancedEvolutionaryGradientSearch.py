import numpy as np


class EnhancedEvolutionaryGradientSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = None

    def approximate_gradient(self, func, x, epsilon=1e-8):
        grad = np.zeros_like(x)
        fx = func(x)
        for i in range(len(x)):
            x_step = np.array(x)
            x_step[i] += epsilon
            grad[i] = (func(x_step) - fx) / epsilon
        return grad

    def __call__(self, func):
        self.dim = len(func.bounds.lb)
        self.f_opt = np.Inf
        self.x_opt = None

        population_size = 20
        mutation_rate = 0.1
        selection_rate = 0.5
        elite_fraction = 0.1
        grad_step = 0.01

        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        while evaluations < self.budget:
            # Select elites and breed new generation
            num_elites = int(elite_fraction * population_size)
            elite_indices = np.argsort(fitness)[:num_elites]
            elites = population[elite_indices]

            # Create new population
            new_population = []
            for _ in range(population_size):
                if np.random.rand() < selection_rate:
                    parents = np.random.choice(num_elites, 2, replace=False)
                    parent1, parent2 = elites[parents[0]], elites[parents[1]]
                    offspring = (parent1 + parent2) / 2
                else:
                    offspring = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)

                # Mutation
                if np.random.rand() < mutation_rate:
                    mutation_vector = np.random.randn(self.dim) * 0.1
                    offspring = offspring + mutation_vector

                offspring = np.clip(offspring, func.bounds.lb, func.bounds.ub)
                new_population.append(offspring)

            new_population = np.array(new_population)

            # Evaluate new population
            new_fitness = np.array([func(ind) for ind in new_population])
            evaluations += population_size

            # Gradient-based local search for elites
            for i in range(num_elites):
                elite = elites[i]
                grad = self.approximate_gradient(func, elite)
                elite_new = elite - grad_step * grad
                elite_new = np.clip(elite_new, func.bounds.lb, func.bounds.ub)
                elite_fitness = func(elite_new)
                evaluations += 1

                if elite_fitness < fitness[elite_indices[i]]:
                    new_population[elite_indices[i]] = elite_new
                    new_fitness[elite_indices[i]] = elite_fitness

            # Update population and fitness
            population = new_population
            fitness = new_fitness

            # Update best solution found
            min_fitness_idx = np.argmin(fitness)
            if fitness[min_fitness_idx] < self.f_opt:
                self.f_opt = fitness[min_fitness_idx]
                self.x_opt = population[min_fitness_idx]

        return self.f_opt, self.x_opt
