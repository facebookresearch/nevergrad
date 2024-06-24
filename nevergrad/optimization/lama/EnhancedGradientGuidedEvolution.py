import numpy as np


class EnhancedGradientGuidedEvolution:
    def __init__(
        self,
        budget,
        dimension=5,
        population_size=30,
        mutation_factor=0.7,
        crossover_prob=0.8,
        local_search_prob=0.15,
        gradient_step_size=0.005,
    ):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.local_search_prob = local_search_prob
        self.gradient_step_size = gradient_step_size  # Step size for gradient approximation

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dimension))
        fitness = np.array([func(individual) for individual in population])
        best_idx = np.argmin(fitness)
        f_opt, x_opt = fitness[best_idx], population[best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            new_population = []
            for i in range(self.population_size):
                # Mutation and crossover using differential evolution strategy
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), -5.0, 5.0)

                # Crossover
                trial = np.where(np.random.rand(self.dimension) < self.crossover_prob, mutant, population[i])
                trial_fitness = func(trial)
                evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    new_population.append(trial)
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt, x_opt = trial_fitness, trial
                else:
                    new_population.append(population[i])

                if evaluations >= self.budget:
                    break

            population = np.array(new_population)

            # Intermittent gradient-based local search
            if np.random.rand() < self.local_search_prob and evaluations + self.dimension <= self.budget:
                for _ in range(int(0.05 * self.budget)):  # Limited local search steps
                    gradient = np.zeros(self.dimension)
                    for d in range(self.dimension):
                        perturb = np.zeros(self.dimension)
                        perturb[d] = self.gradient_step_size
                        f_plus = func(x_opt + perturb)
                        f_minus = func(x_opt - perturb)
                        gradient[d] = (f_plus - f_minus) / (2 * self.gradient_step_size)
                        evaluations += 2

                        if evaluations >= self.budget:
                            break

                    # Update the best solution based on the gradient information
                    x_opt = np.clip(x_opt - 0.01 * gradient, -5.0, 5.0)
                    f_opt = func(x_opt)
                    evaluations += 1

                    if evaluations >= self.budget:
                        break

        return f_opt, x_opt
