import numpy as np


class EGGEO:
    def __init__(
        self,
        budget,
        population_size=50,
        gradient_impact=0.1,
        random_impact=0.05,
        mutation_rate=0.1,
        elitism=2,
    ):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0
        self.gradient_impact = gradient_impact
        self.random_impact = random_impact
        self.mutation_rate = mutation_rate
        self.elitism = elitism  # Number of elites

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        num_evals = self.population_size

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()

        # Evolutionary loop
        while num_evals < self.budget:
            # Selection based on fitness
            sorted_indices = np.argsort(fitness)
            elites_indices = sorted_indices[: self.elitism]
            elites = population[elites_indices]

            new_population = np.zeros_like(population)
            new_population[: self.elitism] = elites.copy()  # Elitism

            for i in range(self.elitism, self.population_size):
                parent1_idx, parent2_idx = np.random.choice(elites_indices, 2, replace=False)
                crossover_point = np.random.randint(self.dimension)

                # Crossover
                child = np.concatenate(
                    [population[parent1_idx][:crossover_point], population[parent2_idx][crossover_point:]]
                )

                # Mutation
                mutation_mask = np.random.rand(self.dimension) < self.mutation_rate
                mutation_values = np.random.uniform(-1, 1, self.dimension)
                child += mutation_mask * mutation_values

                # Gradient guidance
                grad_direction = best_individual - child
                child += self.gradient_impact * grad_direction + self.random_impact * np.random.uniform(
                    -1, 1, self.dimension
                )
                child = np.clip(child, self.lb, self.ub)

                new_population[i] = child

            population = new_population
            fitness = np.array([func(ind) for ind in population])
            num_evals += self.population_size - self.elitism

            # Update best
            current_best_idx = np.argmin(fitness)
            current_best_fitness = fitness[current_best_idx]
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_idx].copy()

        return best_fitness, best_individual
