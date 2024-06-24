import numpy as np


class HybridDifferentialEvolutionWithLocalSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # fixed dimensionality

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

        bounds = np.array([-5.0, 5.0])
        population_size = 20
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover probability

        # Initialize population
        population = np.random.uniform(bounds[0], bounds[1], (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        evaluations = population_size

        # Initialize self-adaptive parameters
        F_values = np.full(population_size, F)
        CR_values = np.full(population_size, CR)

        while evaluations < self.budget:
            new_population = np.zeros_like(population)
            new_fitness = np.zeros(population_size)
            new_F_values = np.zeros(population_size)
            new_CR_values = np.zeros(population_size)

            for i in range(population_size):
                # Adaptation of F and CR
                if np.random.rand() < 0.1:
                    F_values[i] = 0.1 + 0.9 * np.random.rand()
                if np.random.rand() < 0.1:
                    CR_values[i] = np.random.rand()

                # Mutation
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F_values[i] * (b - c), bounds[0], bounds[1])

                # Crossover
                cross_points = np.random.rand(self.dim) < CR_values[i]
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                f_trial = func(trial)
                evaluations += 1

                if f_trial < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial
                    new_F_values[i] = F_values[i]
                    new_CR_values[i] = CR_values[i]

                    # Update the best found solution
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    new_population[i] = population[i]
                    new_fitness[i] = fitness[i]
                    new_F_values[i] = F_values[i]
                    new_CR_values[i] = CR_values[i]

                if evaluations >= self.budget:
                    break

            population, fitness = new_population, new_fitness
            F_values, CR_values = new_F_values, new_CR_values

            # Apply local search on the best solution found so far
            if evaluations < self.budget:
                local_search_budget = int(self.budget * 0.1)  # allocate 10% of the budget for local search
                for _ in range(local_search_budget):
                    perturbation = np.random.normal(0, 0.1, self.dim)
                    local_trial = np.clip(self.x_opt + perturbation, bounds[0], bounds[1])
                    f_local_trial = func(local_trial)
                    evaluations += 1

                    if f_local_trial < self.f_opt:
                        self.f_opt = f_local_trial
                        self.x_opt = local_trial

                    if evaluations >= self.budget:
                        break

        return self.f_opt, self.x_opt
