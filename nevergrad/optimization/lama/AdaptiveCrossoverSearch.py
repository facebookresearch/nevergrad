import numpy as np


class AdaptiveCrossoverSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        # Initialize parameters
        population_size = 20
        crossover_rate = 0.7
        mutation_rate = 0.1
        elitism_rate = 0.1

        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]

        evaluations = population_size

        while evaluations < self.budget:
            new_population = []

            # Elitism: carry forward the best solutions
            elitism_count = int(elitism_rate * population_size)
            elite_indices = np.argsort(fitness)[:elitism_count]
            new_population.extend(population[elite_indices])

            while len(new_population) < population_size:
                # Select parents
                idx1, idx2 = np.random.choice(population_size, 2, replace=False)
                parent1, parent2 = population[idx1], population[idx2]

                # Crossover
                if np.random.rand() < crossover_rate:
                    crossover_point = np.random.randint(1, self.dim)
                    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
                else:
                    child1, child2 = parent1, parent2

                # Mutation
                for child in [child1, child2]:
                    if np.random.rand() < mutation_rate:
                        mutation_idx = np.random.randint(self.dim)
                        child[mutation_idx] = np.random.uniform(self.lb, self.ub)

                    new_population.append(child)
                    if len(new_population) >= population_size:
                        break

            # Evaluate new population
            new_fitness = np.array([func(ind) for ind in new_population])
            evaluations += population_size

            # Update population
            population = np.array(new_population)
            fitness = new_fitness

            # Update best solution
            current_best_idx = np.argmin(fitness)
            current_best_fitness = fitness[current_best_idx]
            if current_best_fitness < self.f_opt:
                self.f_opt = current_best_fitness
                self.x_opt = population[current_best_idx]

        return self.f_opt, self.x_opt
