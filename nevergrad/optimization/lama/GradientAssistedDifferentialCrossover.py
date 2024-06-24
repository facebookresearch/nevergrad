import numpy as np


class GradientAssistedDifferentialCrossover:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        population_size = 100
        elite_size = 10
        mutation_factor = 0.8
        crossover_rate = 0.9

        # Initialize population and evaluate fitness
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        evaluations = population_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        while evaluations < self.budget:
            new_population = []
            for i in range(population_size):
                # Selection for differential evolution operations
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                # Mutation: Differential mutation
                mutant = np.clip(a + mutation_factor * (b - c), self.lb, self.ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < crossover_rate
                trial = np.where(cross_points, mutant, population[i])

                # Gradient assistance in selection
                if np.random.rand() < 0.5:  # With a 50% chance, refine using gradient information
                    grad_direction = trial - population[i]
                    grad_step = 0.1 * grad_direction  # Step size
                    trial = np.clip(population[i] + grad_step, self.lb, self.ub)

                # Selection: Greedy selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    new_population.append(trial)
                    fitness[i] = trial_fitness

                    # Update best found solution
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial

                else:
                    new_population.append(population[i])

            population = np.array(new_population)

            # Elitism: Carry over a few best solutions directly
            elite_indices = np.argsort(fitness)[:elite_size]
            non_elite_population = [pop for idx, pop in enumerate(population) if idx not in elite_indices]
            population = np.vstack(
                (population[elite_indices], non_elite_population[: population_size - elite_size])
            )

        return self.f_opt, self.x_opt
