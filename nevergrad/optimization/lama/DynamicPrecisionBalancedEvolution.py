import numpy as np


class DynamicPrecisionBalancedEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        population_size = 80
        elite_size = 10
        mutation_factor = 0.9
        crossover_probability = 0.85
        recombination_weight = 0.1

        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        evaluations = population_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        while evaluations < self.budget:
            new_population = []
            for i in range(population_size):
                # Differential mutation
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant_vector = a + mutation_factor * (b - c)
                mutant_vector = np.clip(mutant_vector, self.lb, self.ub)

                # Dynamic recombination towards best
                mutant_vector = (1 - recombination_weight) * mutant_vector + recombination_weight * self.x_opt

                # Binomial crossover
                trial_vector = np.array(
                    [
                        (
                            mutant_vector[j]
                            if np.random.rand() < crossover_probability or j == np.random.randint(self.dim)
                            else population[i, j]
                        )
                        for j in range(self.dim)
                    ]
                )

                # Fitness evaluation and selection
                trial_fitness = func(trial_vector)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    new_population.append(trial_vector)
                    fitness[i] = trial_fitness
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial_vector
                else:
                    new_population.append(population[i])

            population = np.array(new_population)

            # Update mutation factor dynamically based on performance
            mutation_factor = np.clip(
                mutation_factor + 0.02 * (self.f_opt / np.median(fitness) - 1), 0.5, 1.0
            )

            # Elite replacement to maintain diversity
            elite_indices = np.argsort(fitness)[:elite_size]
            elite_individuals = population[elite_indices]
            for idx in np.random.choice(range(population_size), elite_size, replace=False):
                population[idx] = elite_individuals[np.random.randint(elite_size)]

        return self.f_opt, self.x_opt
