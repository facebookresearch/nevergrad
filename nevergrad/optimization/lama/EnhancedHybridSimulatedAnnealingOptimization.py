import numpy as np
from scipy.spatial.distance import cdist


class EnhancedHybridSimulatedAnnealingOptimization:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0

        # Parameters for DE
        self.population_size = 100
        self.F_min = 0.4
        self.F_max = 0.9
        self.CR_min = 0.1
        self.CR_max = 0.9

        # PSO Parameters
        self.inertia_weight = 0.9
        self.cognitive_constant = 2.0
        self.social_constant = 2.0

        # Stagnation control
        self.stagnation_threshold = 10
        self.stagnation_counter = 0

        # Elitism
        self.elite_fraction = 0.1

        # Memory Mechanism
        self.memory_size = 10
        self.memory = []

    def _latin_hypercube_sampling(self, n_samples, n_dim):
        l = np.arange(n_samples)
        np.random.shuffle(l)
        sample = l / n_samples + np.random.rand(n_samples) / n_samples
        return sample

    def _initialize_population(self):
        lhs_samples = np.zeros((self.population_size, self.dim))
        for i in range(self.dim):
            lhs_samples[:, i] = self._latin_hypercube_sampling(self.population_size, self.dim)
        lhs_samples = self.lb + lhs_samples * (self.ub - self.lb)
        random_samples = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        population = np.vstack((lhs_samples, random_samples))[: self.population_size]
        return population

    def _simulated_annealing(self, x, func):
        temp = 1.0
        cooling_rate = 0.99
        best_x = x.copy()
        best_f = func(x)
        while temp > 1e-3:
            new_x = np.clip(x + np.random.normal(0, 1, self.dim), self.lb, self.ub)
            new_f = func(new_x)
            if new_f < best_f or np.random.rand() < np.exp((best_f - new_f) / temp):
                best_x = new_x
                best_f = new_f
            temp *= cooling_rate
        return best_x, best_f

    def __call__(self, func):
        # Initialize population
        population = self._initialize_population()
        fitness = np.array([func(ind) for ind in population])
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))

        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)].copy()

        personal_best_positions = population.copy()
        personal_best_fitness = fitness.copy()

        evaluations = self.population_size
        best_fitness_history = [self.f_opt]

        while evaluations < self.budget:
            # Elitism Preservation
            elite_count = int(self.elite_fraction * self.population_size)
            elites = population[np.argsort(fitness)[:elite_count]].copy()
            elite_fitness = np.sort(fitness)[:elite_count].copy()

            for i in range(self.population_size):
                # Select three random vectors a, b, c from population
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                # Adaptive Mutation and Crossover
                F_adaptive = self.F_min + np.random.rand() * (self.F_max - self.F_min)
                CR_adaptive = self.CR_min + np.random.rand() * (self.CR_max - self.CR_min)

                mutant_vector = np.clip(a + F_adaptive * (b - c), self.lb, self.ub)

                trial_vector = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < CR_adaptive:
                        trial_vector[j] = mutant_vector[j]

                f_candidate = func(trial_vector)
                evaluations += 1

                if f_candidate < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = f_candidate

                    if f_candidate < self.f_opt:
                        self.f_opt = f_candidate
                        self.x_opt = trial_vector
                        self.stagnation_counter = 0
                    else:
                        self.stagnation_counter += 1
                else:
                    self.stagnation_counter += 1

                if evaluations >= self.budget:
                    break

                # Update personal best
                if f_candidate < personal_best_fitness[i]:
                    personal_best_positions[i] = trial_vector
                    personal_best_fitness[i] = f_candidate

            # Integrate Elitism
            population[np.argsort(fitness)[-elite_count:]] = elites
            fitness[np.argsort(fitness)[-elite_count:]] = elite_fitness

            # Update velocities and positions (PSO component)
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            velocities = (
                self.inertia_weight * velocities
                + self.cognitive_constant * r1 * (personal_best_positions - population)
                + self.social_constant * r2 * (self.x_opt - population)
            )
            population = np.clip(population + velocities, self.lb, self.ub)

            # Evaluate new population
            for i in range(self.population_size):
                f_candidate = func(population[i])
                evaluations += 1

                if f_candidate < fitness[i]:
                    fitness[i] = f_candidate
                    if f_candidate < self.f_opt:
                        self.f_opt = f_candidate
                        self.x_opt = population[i]
                        self.stagnation_counter = 0
                    else:
                        self.stagnation_counter += 1
                else:
                    self.stagnation_counter += 1

                if evaluations >= self.budget:
                    break

            # Store best fitness
            best_fitness_history.append(self.f_opt)

            # Adaptive Parameter Adjustment based on Stagnation Counter
            if self.stagnation_counter > self.stagnation_threshold:
                self.F_max = min(1.0, self.F_max + 0.1)
                self.CR_max = min(1.0, self.CR_max + 0.1)
                self.stagnation_counter = 0
            else:
                self.F_max = max(self.F_min, self.F_max - 0.1)
                self.CR_max = max(self.CR_min, self.CR_max - 0.1)

            # Adjust inertia weight dynamically
            self.inertia_weight = 0.4 + 0.5 * (self.budget - evaluations) / self.budget

            # Adjust population size dynamically based on performance
            if self.stagnation_counter > self.stagnation_threshold * 2:
                new_population_size = min(self.population_size + 10, 200)
                if new_population_size > self.population_size:
                    new_individuals = np.random.uniform(
                        self.lb, self.ub, (new_population_size - self.population_size, self.dim)
                    )
                    population = np.vstack((population, new_individuals))
                    new_velocities = np.random.uniform(
                        -1, 1, (new_population_size - self.population_size, self.dim)
                    )
                    velocities = np.vstack((velocities, new_velocities))
                    new_fitness = np.array([func(ind) for ind in new_individuals])
                    fitness = np.hstack((fitness, new_fitness))
                    personal_best_positions = np.vstack((personal_best_positions, new_individuals))
                    personal_best_fitness = np.hstack((personal_best_fitness, new_fitness))
                    self.population_size = new_population_size
                    evaluations += new_population_size - self.population_size

            # Memory mechanism
            if len(self.memory) < self.memory_size:
                self.memory.append(self.x_opt)
            else:
                worst_mem_idx = np.argmin([func(mem) for mem in self.memory])
                self.memory[worst_mem_idx] = self.x_opt

            # Enhanced Local Search using Simulated Annealing
            if self.stagnation_counter > self.stagnation_threshold * 3:
                for i, mem in enumerate(self.memory):
                    local_candidate, f_local_candidate = self._simulated_annealing(mem, func)
                    evaluations += 1

                    if f_local_candidate < func(mem):
                        self.memory[i] = local_candidate
                        if f_local_candidate < self.f_opt:
                            self.f_opt = f_local_candidate
                            self.x_opt = local_candidate
                            self.stagnation_counter = 0

                    if evaluations >= self.budget:
                        break

        return self.f_opt, self.x_opt
