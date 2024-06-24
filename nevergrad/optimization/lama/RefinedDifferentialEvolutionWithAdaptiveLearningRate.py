import numpy as np


class RefinedDifferentialEvolutionWithAdaptiveLearningRate:
    def __init__(self, budget=10000):
        self.budget = budget
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.dim = 5
        self.population_size = 100  # Population size
        self.initial_F = 0.5  # Differential evolution mutation factor
        self.initial_CR = 0.9  # Differential evolution crossover rate
        self.self_adaptive_rate = 0.1  # Self-adaptive rate for parameters
        self.elite_rate = 0.1  # Elite retention rate
        self.memory_size = 20  # Memory size for adaptive parameters
        self.adaptive_phase_ratio = 0.7  # Budget ratio for evolutionary phase
        self.local_search_rate = 0.2  # Probability for local search
        self.alpha = 0.6  # Differential weight for local search
        self.adaptive_learning_rate = 0.01  # Learning rate for adaptive mutation factor and crossover rate

    def __call__(self, func):
        def clip_bounds(candidate):
            return np.clip(candidate, self.lower_bound, self.upper_bound)

        def initialize_population():
            population = np.random.uniform(
                self.lower_bound, self.upper_bound, (self.population_size, self.dim)
            )
            fitness = np.array([func(ind) for ind in population])
            return population, fitness

        population, fitness = initialize_population()
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))

        personal_best_positions = np.copy(population)
        personal_best_fitness = np.copy(fitness)

        best_index = np.argmin(fitness)
        best_position = population[best_index]
        best_value = fitness[best_index]

        self.eval_count = self.population_size

        memory_F = np.full(self.memory_size, self.initial_F)
        memory_CR = np.full(self.memory_size, self.initial_CR)
        memory_idx = 0

        def local_search(position):
            step_size = 0.1  # Step size for local search
            candidate = position + np.random.uniform(-step_size, step_size, position.shape)
            return clip_bounds(candidate)

        def adapt_parameters():
            idx = np.random.randint(0, self.memory_size)
            adaptive_F = memory_F[idx] + self.adaptive_learning_rate * np.random.randn()
            adaptive_CR = memory_CR[idx] + self.adaptive_learning_rate * np.random.randn()
            return np.clip(adaptive_F, 0.1, 1.0), np.clip(adaptive_CR, 0.1, 1.0)

        def evolutionary_phase():
            nonlocal best_value, best_position, memory_idx
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)

            elite_count = int(self.elite_rate * self.population_size)
            sorted_indices = np.argsort(fitness)
            elites = population[sorted_indices[:elite_count]]
            new_population[:elite_count] = elites
            new_fitness[:elite_count] = fitness[sorted_indices[:elite_count]]

            for i in range(elite_count, self.population_size):
                if self.eval_count >= self.budget:
                    break
                a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                F, CR = adapt_parameters()
                mutant = clip_bounds(a + F * (b - c))
                trial = np.copy(population[i])
                for d in range(self.dim):
                    if np.random.rand() < CR:
                        trial[d] = mutant[d]

                if np.random.rand() < self.local_search_rate:
                    candidate = local_search(trial)
                else:
                    candidate = trial

                candidate = clip_bounds(candidate)
                candidate_value = func(candidate)
                self.eval_count += 1

                if candidate_value < fitness[i]:
                    new_population[i] = candidate
                    new_fitness[i] = candidate_value

                    memory_F[memory_idx] = F
                    memory_CR[memory_idx] = CR
                    memory_idx = (memory_idx + 1) % self.memory_size

                if candidate_value < best_value:
                    best_value = candidate_value
                    best_position = candidate

            return new_population, new_fitness

        def swarm_phase():
            nonlocal best_value, best_position
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)

            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocities[i] = (
                    0.5 * velocities[i]
                    + 1.5 * r1 * (personal_best_positions[i] - population[i])
                    + 1.5 * r2 * (best_position - population[i])
                )
                new_population[i] = clip_bounds(population[i] + velocities[i])
                new_fitness[i] = func(new_population[i])
                self.eval_count += 1

                if new_fitness[i] < personal_best_fitness[i]:
                    personal_best_fitness[i] = new_fitness[i]
                    personal_best_positions[i] = new_population[i]

                if new_fitness[i] < best_value:
                    best_value = new_fitness[i]
                    best_position = new_population[i]

            return new_population, new_fitness

        while self.eval_count < self.budget:
            if self.eval_count < self.adaptive_phase_ratio * self.budget:
                population, fitness = evolutionary_phase()
            else:
                population, fitness = swarm_phase()

        return best_value, best_position


# Example usage:
# func = SomeBlackBoxFunction()  # The black box function to be optimized
# optimizer = RefinedDifferentialEvolutionWithAdaptiveLearningRate(budget=10000)
# best_value, best_position = optimizer(func)
