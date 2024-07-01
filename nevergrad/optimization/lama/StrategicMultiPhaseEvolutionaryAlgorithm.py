import numpy as np


class StrategicMultiPhaseEvolutionaryAlgorithm:
    def __init__(self, budget, dimension=5, lower_bound=-5.0, upper_bound=5.0, population_size=50):
        self.budget = budget
        self.dimension = dimension
        self.bounds = {"lb": lower_bound, "ub": upper_bound}
        self.population_size = population_size
        self.mutation_rate = 0.1  # Initial mutation rate
        self.crossover_probability = 0.8  # Probability of crossover
        self.elitism = True  # Enable elitism

    def mutate(self, individual, phase):
        """Apply Gaussian mutation based on the phase of the algorithm."""
        if phase < 0.3:
            scale = 0.5  # High exploration in the early phase
        elif phase < 0.6:
            scale = 0.2  # Focused exploration in the mid phase
        else:
            scale = 0.05  # Fine-tuning in the late phase

        mutation = np.random.normal(0, scale, self.dimension)
        mutant = individual + mutation
        return np.clip(mutant, self.bounds["lb"], self.bounds["ub"])

    def crossover(self, parent1, parent2):
        """Simulated binary crossover, for better offspring production."""
        alpha = np.random.uniform(-0.5, 1.5, self.dimension)
        offspring = alpha * parent1 + (1 - alpha) * parent2
        return np.clip(offspring, self.bounds["lb"], self.bounds["ub"])

    def select(self, population, fitness, offspring, offspring_fitness):
        """Selects the next generation using elitism and tournament selection."""
        combined_population = np.vstack((population, offspring))
        combined_fitness = np.concatenate((fitness, offspring_fitness))
        indices = np.argsort(combined_fitness)

        if self.elitism:
            best_indices = indices[: self.population_size]
        else:
            # Random selection with preference to lower fitness
            probabilities = 1 / (1 + np.exp(combined_fitness - np.median(combined_fitness)))
            best_indices = np.random.choice(
                len(combined_population), size=self.population_size, p=probabilities / np.sum(probabilities)
            )

        return combined_population[best_indices], combined_fitness[best_indices]

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(
            self.bounds["lb"], self.bounds["ub"], (self.population_size, self.dimension)
        )
        fitness = np.array([func(individual) for individual in population])
        f_opt = np.min(fitness)
        x_opt = population[np.argmin(fitness)]

        # Evolutionary loop
        for iteration in range(self.budget // self.population_size):
            phase = iteration / (self.budget / self.population_size)
            offspring = []
            offspring_fitness = []

            # Generate offspring
            for idx in range(self.population_size):
                # Mutation
                mutant = self.mutate(population[idx], phase)

                # Crossover
                if np.random.rand() < self.crossover_probability:
                    partner_idx = np.random.randint(self.population_size)
                    child = self.crossover(mutant, population[partner_idx])
                else:
                    child = mutant

                # Evaluate
                child_fitness = func(child)
                offspring.append(child)
                offspring_fitness.append(child_fitness)

            # Selection
            offspring = np.array(offspring)
            offspring_fitness = np.array(offspring_fitness)
            population, fitness = self.select(population, fitness, offspring, offspring_fitness)

            # Update best found solution
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < f_opt:
                f_opt = fitness[min_idx]
                x_opt = population[min_idx]

        return f_opt, x_opt
