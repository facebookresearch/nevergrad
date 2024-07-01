import numpy as np


class CooperativeAdaptiveCulturalSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0

    def local_search(self, x, func, step_size=0.1, max_iter=10):
        """Adaptive local search around a point."""
        best_x = x.copy()
        best_f = func(x)
        for _ in range(max_iter):
            perturbation = np.random.normal(0, step_size, self.dim)
            new_x = np.clip(x + perturbation, self.lb, self.ub)
            new_f = func(new_x)

            if new_f < best_f:
                best_x = new_x
                best_f = new_f
                step_size *= 0.9  # decrease step size if improvement is found
            else:
                step_size *= 1.1  # increase step size if no improvement

        return best_x, best_f

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        population_size = 50
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = len(fitness)

        # Initialize strategy parameters for each individual
        strategy_params = np.random.uniform(0.1, 0.3, (population_size, self.dim))

        # Initialize cultural component
        knowledge_base = {
            "best_solution": None,
            "best_fitness": np.inf,
            "mean_position": np.mean(population, axis=0),
            "standard_deviation": np.std(population, axis=0),
        }

        while evaluations < self.budget:
            for i in range(population_size):
                if evaluations >= self.budget:
                    break

                # Mutation using Evolution Strategy
                strategy_noise = np.random.normal(0, strategy_params[i], self.dim)
                trial_vector = population[i] + strategy_noise
                trial_vector = np.clip(trial_vector, self.lb, self.ub)

                # Evaluate trial vector
                trial_fitness = func(trial_vector)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

                    # Adapt strategy parameters
                    strategy_params[i] *= np.exp(0.1 * (np.random.rand(self.dim) - 0.5))

                    # Update cultural knowledge
                    if trial_fitness < knowledge_base["best_fitness"]:
                        knowledge_base["best_solution"] = trial_vector
                        knowledge_base["best_fitness"] = trial_fitness
                    knowledge_base["mean_position"] = np.mean(population, axis=0)
                    knowledge_base["standard_deviation"] = np.std(population, axis=0)

                    # Update global best
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial_vector

                # Apply local search on some individuals
                if np.random.rand() < 0.3:
                    local_best_x, local_best_f = self.local_search(population[i], func)
                    evaluations += 10  # Assuming local search uses 10 evaluations
                    if local_best_f < fitness[i]:
                        population[i] = local_best_x
                        fitness[i] = local_best_f
                        if local_best_f < self.f_opt:
                            self.f_opt = local_best_f
                            self.x_opt = local_best_x

            # Cultural-based guidance: periodically update population based on cultural knowledge
            if evaluations % (population_size * 2) == 0:  # More infrequent updates
                if knowledge_base["best_solution"] is None:
                    continue  # Skip if no best solution has been found yet

                # Adjust cultural shift influence dynamically based on fitness diversity
                fitness_std = np.std(fitness)
                cultural_influence = 0.1 + (0.2 * fitness_std / (np.mean(fitness) + 1e-9))
                cultural_shift = (
                    knowledge_base["best_solution"] - knowledge_base["mean_position"]
                ) * cultural_influence

                # Cooperative cultural influence updates with standard deviation
                for i in range(population_size):
                    cooperation_factor = np.random.rand()
                    shift = cooperation_factor * cultural_shift + (1 - cooperation_factor) * knowledge_base[
                        "standard_deviation"
                    ] * np.random.normal(0, 0.1, self.dim)
                    population[i] = np.clip(population[i] + shift, self.lb, self.ub)

                fitness = np.array([func(ind) for ind in population])
                evaluations += population_size

                # Reinitialize strategy parameters for new individuals
                strategy_params = np.random.uniform(0.1, 0.3, (population_size, self.dim))

        return self.f_opt, self.x_opt
