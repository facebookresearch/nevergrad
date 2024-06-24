import numpy as np


class GuidedEvolutionStrategy:
    def __init__(
        self, budget, dimension=5, population_size=50, sigma=0.5, learning_rate=0.7, mutation_probability=0.1
    ):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.mutation_probability = mutation_probability

    def __call__(self, func):
        # Initialize the population and the best solution found
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dimension))
        f_opt = np.Inf
        x_opt = None

        # Track the number of function evaluations
        evaluations = 0

        while evaluations < self.budget:
            # Evaluate the current population
            fitness = np.array([func(individual) for individual in population])
            evaluations += len(population)

            # Find the best individual in the current population
            best_index = np.argmin(fitness)
            best_fitness = fitness[best_index]
            best_individual = population[best_index]

            # Update the global best if found a new best
            if best_fitness < f_opt:
                f_opt = best_fitness
                x_opt = best_individual

            # Generate new individuals by mutation and recombination
            new_population = []
            for _ in range(self.population_size):
                if np.random.rand() < self.mutation_probability:
                    # Mutation: add Gaussian noise
                    individual = best_individual + np.random.normal(0, self.sigma, self.dimension)
                else:
                    # Recombination: crossover between two random individuals
                    parents = population[np.random.choice(self.population_size, 2, replace=False)]
                    crossover_point = np.random.randint(0, self.dimension)
                    individual = np.concatenate((parents[0][:crossover_point], parents[1][crossover_point:]))

                # Make sure the individuals are within bounds
                individual = np.clip(individual, -5.0, 5.0)
                new_population.append(individual)

            population = np.array(new_population)

            # Reduce the mutation size over time to allow fine-tuning
            self.sigma *= self.learning_rate

        return f_opt, x_opt
