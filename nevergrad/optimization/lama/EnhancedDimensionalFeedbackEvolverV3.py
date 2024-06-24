import numpy as np


class EnhancedDimensionalFeedbackEvolverV3:
    def __init__(
        self,
        budget,
        dimension=5,
        lower_bound=-5.0,
        upper_bound=5.0,
        population_size=350,
        elite_fraction=0.1,
        mutation_intensity=0.025,
        crossover_probability=0.65,
        feedback_factor=0.6,
        exploration_increment=0.05,
    ):
        self.budget = budget
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population_size = population_size
        self.num_elites = int(population_size * elite_fraction)
        self.mutation_intensity = mutation_intensity
        self.crossover_probability = crossover_probability
        self.feedback_factor = feedback_factor
        self.exploration_increment = exploration_increment

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dimension))

    def evaluate_fitness(self, func, population):
        return np.array([func(ind) for ind in population])

    def select_elites(self, population, fitness):
        elite_indices = np.argsort(fitness)[: self.num_elites]
        return population[elite_indices], fitness[elite_indices]

    def mutate(self, individual):
        mutation = np.random.normal(0, self.mutation_intensity, self.dimension)
        return np.clip(individual + mutation, self.lower_bound, self.upper_bound)

    def adaptive_crossover(self, parent1, parent2):
        alpha = np.random.rand()
        return alpha * parent1 + (1 - alpha) * parent2

    def reproduce(self, elites, elite_fitness, previous_population=None):
        new_population = np.empty((self.population_size, self.dimension))
        previous_best = elites[np.argmin(elite_fitness)]

        for i in range(self.population_size):
            if np.random.rand() < self.crossover_probability:
                parents = np.random.choice(self.num_elites, 2, replace=False)
                child = self.adaptive_crossover(elites[parents[0]], elites[parents[1]])
            else:
                child = elites[np.random.choice(self.num_elites)]
            child = self.mutate(child)
            if previous_population is not None:
                feedback_vector = self.feedback_factor * (previous_best - previous_population[i])
                child = np.clip(child + feedback_vector, self.lower_bound, self.upper_bound)
            new_population[i] = child

        # Ensuring the best previous individual is maintained
        new_population[0] = previous_best
        return new_population

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate_fitness(func, population)
        previous_population = None

        best_individual = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)
        evaluations = self.population_size

        while evaluations < self.budget:
            elites, elite_fitness = self.select_elites(population, fitness)
            population = self.reproduce(elites, elite_fitness, previous_population)
            fitness = self.evaluate_fitness(func, population)

            current_best_fitness = np.min(fitness)
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[np.argmin(fitness)]

            previous_population = population
            evaluations += self.population_size

            # Incrementally increase mutation intensity to enhance exploration over time
            self.mutation_intensity += self.exploration_increment * (evaluations / self.budget)

        return best_fitness, best_individual
