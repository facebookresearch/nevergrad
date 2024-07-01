import numpy as np


class DHDGE:
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

        # Adaptive parameters
        mutation_factor = 0.8
        crossover_rate = 0.7
        gradient_learning_rate = 0.1

        while num_evals < self.budget:
            # Gradient estimation for top performers
            elite_idxs = np.argsort(fitness)[: population_size // 5]
            gradients = np.zeros_like(population)

            for idx in elite_idxs:
                perturbation = np.random.normal(
                    0, 0.1 * (self.upper_bound - self.lower_bound), self.dimension
                )
                perturbed_individual = np.clip(
                    population[idx] + perturbation, self.lower_bound, self.upper_bound
                )
                perturbed_fitness = func(perturbed_individual)
                num_evals += 1

                if num_evals >= self.budget:
                    break

                gradient = (perturbed_fitness - fitness[idx]) / (perturbation + 1e-8)
                gradients[idx] = -gradient

            for i in range(population_size):
                if num_evals >= self.budget:
                    break

                # Mutation with gradient influence
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = (
                    population[a]
                    + mutation_factor * (population[b] - population[c])
                    + gradient_learning_rate * gradients[a]
                )
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
