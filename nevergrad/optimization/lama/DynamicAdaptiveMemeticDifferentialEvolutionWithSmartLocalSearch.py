import numpy as np


class DynamicAdaptiveMemeticDifferentialEvolutionWithSmartLocalSearch:
    def __init__(
        self,
        budget=10000,
        pop_size=50,
        init_F=0.8,
        init_CR=0.9,
        elite_fraction=0.1,
        local_search_budget_ratio=0.1,
    ):
        self.budget = budget
        self.pop_size = pop_size
        self.init_F = init_F
        self.init_CR = init_CR
        self.elite_fraction = elite_fraction
        self.local_search_budget_ratio = local_search_budget_ratio
        self.dim = 5  # As stated, dimensionality is 5
        self.bounds = (-5.0, 5.0)  # Bounds are given as [-5.0, 5.0]

    def smart_local_search(self, x, func, budget):
        best_x = x.copy()
        best_f = func(x)
        for _ in range(budget):
            perturbation = np.random.normal(0, 0.1, self.dim)  # Using Gaussian perturbation
            candidate_x = np.clip(x + perturbation, self.bounds[0], self.bounds[1])
            candidate_f = func(candidate_x)
            if candidate_f < best_f:
                best_f = candidate_f
                best_x = candidate_x
        return best_x, best_f

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.eval_count = self.pop_size

        # Differential weights and crossover probabilities for each individual
        F_values = np.full(self.pop_size, self.init_F)
        CR_values = np.full(self.pop_size, self.init_CR)

        local_search_budget = int(self.budget * self.local_search_budget_ratio)
        global_search_budget = self.budget - local_search_budget
        local_search_budget_per_individual = local_search_budget // self.pop_size

        while self.eval_count < global_search_budget:
            for i in range(self.pop_size):
                # Mutation
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                F = F_values[i]
                mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])

                # Crossover
                CR = CR_values[i]
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Incorporate additional crossover mechanism
                if np.random.rand() < 0.3:  # 30% chance to apply blend crossover
                    partner_idx = np.random.choice([idx for idx in range(self.pop_size) if idx != i])
                    partner = population[partner_idx]
                    trial = 0.5 * (trial + partner)
                    trial = np.clip(trial, self.bounds[0], self.bounds[1])

                # Selection
                f_trial = func(trial)
                self.eval_count += 1
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                    # Self-adapting parameters
                    F_values[i] = F * 1.1 if F < 1 else F
                    CR_values[i] = CR * 1.1 if CR < 1 else CR
                else:
                    F_values[i] = F * 0.9 if F > 0 else F
                    CR_values[i] = CR * 0.9 if CR > 0 else CR

                if self.eval_count >= global_search_budget:
                    break

            # Adaptive Elitism: Adjust elite size based on convergence rate
            elite_size = max(
                1, int(self.elite_fraction * self.pop_size * (1 - self.eval_count / global_search_budget))
            )
            elite_indices = np.argsort(fitness)[:elite_size]
            elite_population = population[elite_indices]
            elite_fitness = fitness[elite_indices]

            # Update population with new solutions while preserving elite
            non_elite_indices = np.argsort(fitness)[elite_size:]
            for idx in non_elite_indices:
                if self.eval_count >= global_search_budget:
                    break
                x_new = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
                f_new = func(x_new)
                self.eval_count += 1
                if f_new < fitness[idx]:
                    fitness[idx] = f_new
                    population[idx] = x_new

            population[:elite_size] = elite_population
            fitness[:elite_size] = elite_fitness

        # Perform smart local search on the best individuals
        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break
            local_budget = min(local_search_budget_per_individual, self.budget - self.eval_count)
            new_x, new_f = self.smart_local_search(population[i], func, local_budget)
            self.eval_count += local_budget
            if new_f < fitness[i]:
                fitness[i] = new_f
                population[i] = new_x

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        return self.f_opt, self.x_opt
