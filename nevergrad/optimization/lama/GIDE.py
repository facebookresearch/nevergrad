import numpy as np


class GIDE:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population_size = 100
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        num_evals = population_size

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()

        # Adaptation parameters
        mutation_factor = 0.8
        crossover_rate = 0.9
        learning_rate = 0.1

        while num_evals < self.budget:
            # Estimating the gradient based on the best individuals
            gradients = np.zeros((population_size, self.dimension))
            elite_idx = np.argsort(fitness)[: population_size // 5]

            for i in elite_idx:
                perturb = np.random.normal(0, 0.1, self.dimension)
                perturbed_individual = np.clip(population[i] + perturb, self.lower_bound, self.upper_bound)
                perturbed_fitness = func(perturbed_individual)
                num_evals += 1

                if num_evals >= self.budget:
                    break

                gradient_estimate = (perturbed_fitness - fitness[i]) / perturb
                gradients[i] = -gradient_estimate

            for i in range(population_size):
                if num_evals >= self.budget:
                    break

                # Mutation
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b = np.random.choice(idxs, 2, replace=False)
                mutant = population[i] + mutation_factor * (population[a] - population[b])

                # Gradient descent direction update
                mutant += learning_rate * gradients[i]
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                cross_points = np.random.rand(self.dimension) < crossover_rate
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                num_evals += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial.copy()

        return best_fitness, best_individual
