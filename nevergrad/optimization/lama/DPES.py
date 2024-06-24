import numpy as np


class DPES:
    def __init__(self, budget, population_size=50, initial_step=0.5, step_reduction=0.98, learning_rate=0.1):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0
        self.initial_step = initial_step
        self.step_reduction = step_reduction
        self.learning_rate = learning_rate

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        num_evals = self.population_size

        step_size = self.initial_step
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()

        while num_evals < self.budget:
            new_population = np.zeros_like(population)

            # Evolve each individual
            for i in range(self.population_size):
                perturbation = np.random.randn(self.dimension) * step_size
                candidate = population[i] + perturbation
                candidate = np.clip(candidate, self.lb, self.ub)
                candidate_fitness = func(candidate)
                num_evals += 1

                # Selection process
                if candidate_fitness < fitness[i]:
                    new_population[i] = candidate
                    fitness[i] = candidate_fitness
                    # Update best found solution
                    if candidate_fitness < best_fitness:
                        best_fitness = candidate_fitness
                        best_individual = candidate.copy()
                else:
                    new_population[i] = population[i]

            # Adaptive step-size control
            step_size *= self.step_reduction

            # Learning phase - update based on the best individual
            for j in range(self.population_size):
                if np.random.rand() < 0.5:  # Learning probability
                    direction = best_individual - population[j]
                    new_population[j] += self.learning_rate * direction
                    new_population[j] = np.clip(new_population[j], self.lb, self.ub)
                    # Evaluate the new individual
                    new_fitness = func(new_population[j])
                    num_evals += 1
                    if new_fitness < fitness[j]:
                        population[j] = new_population[j]
                        fitness[j] = new_fitness
                        if new_fitness < best_fitness:
                            best_fitness = new_fitness
                            best_individual = new_population[j].copy()
                    else:
                        population[j] = new_population[j]

        return best_fitness, best_individual
