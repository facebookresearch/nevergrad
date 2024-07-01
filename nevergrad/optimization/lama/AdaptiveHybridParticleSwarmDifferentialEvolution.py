import numpy as np


class AdaptiveHybridParticleSwarmDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.dim = 5
        self.population_size = 50
        self.initial_F = 0.8  # Differential weight
        self.initial_CR = 0.9  # Crossover probability
        self.elite_rate = 0.2  # Elite rate to maintain a portion of elites
        self.local_search_rate = 0.1  # Probability for local search
        self.memory_size = 5  # Memory size for self-adaptation
        self.w = 0.5  # Inertia weight for PSO
        self.c1 = 1.5  # Cognitive coefficient for PSO
        self.c2 = 1.5  # Social coefficient for PSO

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        velocities = np.zeros((self.population_size, self.dim))

        # Initialize personal bests
        personal_best_positions = np.copy(population)
        personal_best_fitness = np.copy(fitness)

        # Initialize global best
        best_index = np.argmin(fitness)
        best_position = population[best_index]
        best_value = fitness[best_index]

        self.eval_count = self.population_size

        memory_F = np.full(self.memory_size, self.initial_F)
        memory_CR = np.full(self.memory_size, self.initial_CR)
        memory_idx = 0

        def local_search(position):
            # Simple local search strategy
            step_size = 0.1
            candidate = position + np.random.uniform(-step_size, step_size, position.shape)
            candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
            return candidate

        def adapt_parameters():
            # Self-adaptive strategy for F and CR with memory
            idx = np.random.randint(0, self.memory_size)
            adaptive_F = memory_F[idx] + (0.1 * np.random.rand() - 0.05)
            adaptive_CR = memory_CR[idx] + (0.1 * np.random.rand() - 0.05)
            return np.clip(adaptive_F, 0.5, 1.0), np.clip(adaptive_CR, 0.5, 1.0)

        while self.eval_count < self.budget:
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)

            # Sort population by fitness and maintain elites
            elite_count = int(self.elite_rate * self.population_size)
            sorted_indices = np.argsort(fitness)
            elites = population[sorted_indices[:elite_count]]
            new_population[:elite_count] = elites
            new_fitness[:elite_count] = fitness[sorted_indices[:elite_count]]

            for i in range(elite_count, self.population_size):
                if self.eval_count >= self.budget:
                    break
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                F, CR = adapt_parameters()
                mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
                trial = np.copy(population[i])
                for d in range(self.dim):
                    if np.random.rand() < CR:
                        trial[d] = mutant[d]

                if np.random.rand() < self.local_search_rate:
                    candidate = local_search(trial)
                else:
                    candidate = trial

                candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                candidate_value = func(candidate)
                self.eval_count += 1

                if candidate_value < fitness[i]:
                    new_population[i] = candidate
                    new_fitness[i] = candidate_value

                    # Update memory
                    memory_F[memory_idx] = F
                    memory_CR[memory_idx] = CR
                    memory_idx = (memory_idx + 1) % self.memory_size

                if candidate_value < best_value:
                    best_value = candidate_value
                    best_position = candidate

            # PSO update for non-elite particles
            for i in range(elite_count, self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (personal_best_positions[i] - population[i])
                    + self.c2 * r2 * (best_position - population[i])
                )
                new_population[i] = np.clip(population[i] + velocities[i], self.lower_bound, self.upper_bound)
                new_fitness[i] = func(new_population[i])
                self.eval_count += 1

                # Update personal bests
                if new_fitness[i] < personal_best_fitness[i]:
                    personal_best_fitness[i] = new_fitness[i]
                    personal_best_positions[i] = new_population[i]

                # Update global best
                if new_fitness[i] < best_value:
                    best_value = new_fitness[i]
                    best_position = new_population[i]

            # Update population and fitness
            population = new_population
            fitness = new_fitness

        return best_value, best_position


# Example usage:
# func = SomeBlackBoxFunction()  # The black box function to be optimized
# optimizer = AdaptiveHybridParticleSwarmDifferentialEvolution(budget=10000)
# best_value, best_position = optimizer(func)
