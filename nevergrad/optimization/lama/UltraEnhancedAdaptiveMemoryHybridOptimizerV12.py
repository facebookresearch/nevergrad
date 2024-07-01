import numpy as np
from scipy.optimize import minimize


class UltraEnhancedAdaptiveMemoryHybridOptimizerV12:
    def __init__(
        self,
        budget=10000,
        init_pop_size=50,
        min_pop_size=20,
        init_F=0.8,
        init_CR=0.9,
        w=0.5,
        c1=1.5,
        c2=1.5,
        local_search_budget_ratio=0.2,
        adaptive_blend_chance=0.3,
        max_memory_size=50,
        early_stop_threshold=0.05,
        enhanced_blend_weight=0.7,
    ):
        self.budget = budget
        self.init_pop_size = init_pop_size
        self.min_pop_size = min_pop_size
        self.init_F = init_F
        self.init_CR = init_CR
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.local_search_budget_ratio = local_search_budget_ratio
        self.adaptive_blend_chance = adaptive_blend_chance
        self.max_memory_size = max_memory_size
        self.early_stop_threshold = early_stop_threshold
        self.enhanced_blend_weight = enhanced_blend_weight
        self.dim = 5
        self.bounds = (-5.0, 5.0)
        self.eval_count = 0
        self.memory = []

    def local_search(self, x, func, budget):
        result = minimize(
            func, x, method="L-BFGS-B", bounds=[self.bounds] * self.dim, options={"maxiter": budget}
        )
        self.eval_count += result.nfev
        return result.x, result.fun

    def adaptive_parameters(self, successful_steps):
        if len(successful_steps) > 0:
            avg_F, avg_CR = np.mean(successful_steps, axis=0)
            return max(0.1, avg_F), max(0.1, avg_CR)
        else:
            return self.init_F, self.init_CR

    def __call__(self, func):
        # Initialize population and velocities for PSO
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.init_pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.eval_count = self.init_pop_size
        velocities = np.random.uniform(-1, 1, (self.init_pop_size, self.dim))

        # Differential weights and crossover probabilities
        F_values = np.full(self.init_pop_size, self.init_F)
        CR_values = np.full(self.init_pop_size, self.init_CR)

        # Initialize the best known positions
        p_best = population.copy()
        p_best_fitness = fitness.copy()
        g_best = population[np.argmin(fitness)]
        g_best_fitness = np.min(fitness)

        local_search_budget = int(self.budget * self.local_search_budget_ratio)
        global_search_budget = self.budget - local_search_budget
        local_search_budget_per_individual = local_search_budget // max(
            self.min_pop_size, 1
        )  # ensure non-zero division

        current_pop_size = self.init_pop_size
        successful_steps = []

        no_improvement_count = 0
        best_fitness_history = []
        max_stagnant_iterations = int(self.early_stop_threshold * global_search_budget)

        while self.eval_count < global_search_budget:
            for i in range(current_pop_size):
                # Adapt parameters
                F, CR = self.adaptive_parameters(successful_steps)

                # PSO update
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (p_best[i] - population[i])
                    + self.c2 * r2 * (g_best - population[i])
                )
                population[i] = np.clip(population[i] + velocities[i], self.bounds[0], self.bounds[1])

                # Mutation
                idxs = [idx for idx in range(current_pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])

                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Incorporate adaptive blending crossover mechanism
                if np.random.rand() < self.adaptive_blend_chance:
                    partner_idx = np.random.choice(idxs)
                    partner = population[partner_idx]
                    trial = self.enhanced_blend_weight * trial + (1 - self.enhanced_blend_weight) * partner
                    trial = np.clip(trial, self.bounds[0], self.bounds[1])

                # Selection
                f_trial = func(trial)
                self.eval_count += 1
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                    successful_steps.append((F, CR))
                    # Limit the memory size
                    if len(successful_steps) > self.max_memory_size:
                        successful_steps.pop(0)
                    # Self-adapting parameters
                    F_values[i] = min(F * 1.1, 1.0)
                    CR_values[i] = min(CR * 1.1, 1.0)
                else:
                    F_values[i] = max(F * 0.9, 0.1)
                    CR_values[i] = max(CR * 0.9, 0.1)

                # Update personal best
                if fitness[i] < p_best_fitness[i]:
                    p_best[i] = population[i]
                    p_best_fitness[i] = fitness[i]

                # Update global best
                if fitness[i] < g_best_fitness:
                    g_best = population[i]
                    g_best_fitness = fitness[i]
                    no_improvement_count = 0  # reset no improvement count
                else:
                    no_improvement_count += 1

                if self.eval_count >= global_search_budget:
                    break

            # Dynamic population resizing based on performance
            if no_improvement_count >= (current_pop_size / 2):
                current_pop_size = max(current_pop_size - 1, self.min_pop_size)
                population = population[:current_pop_size]
                fitness = fitness[:current_pop_size]
                velocities = velocities[:current_pop_size]
                F_values = F_values[:current_pop_size]
                CR_values = CR_values[:current_pop_size]
                p_best = p_best[:current_pop_size]
                p_best_fitness = p_best_fitness[:current_pop_size]
                no_improvement_count = 0

            # Early stopping based on stagnant improvements
            best_fitness_history.append(g_best_fitness)
            if len(best_fitness_history) > max_stagnant_iterations:
                recent_improvement = np.diff(best_fitness_history[-max_stagnant_iterations:])
                if np.all(recent_improvement == 0):
                    break

        # Enhanced elitism: Ensure the best solutions are always retained and perform an elite local search
        elite_individuals = np.argsort(fitness)[: self.min_pop_size]
        elite_population = population[elite_individuals]
        elite_fitness = fitness[elite_individuals]

        # Perform local search on the best individuals, with a bias towards the global best
        for i in range(self.min_pop_size):
            if self.eval_count >= self.budget:
                break
            local_budget = min(local_search_budget_per_individual, self.budget - self.eval_count)
            if np.random.rand() < 0.5:  # 50% chance to do local search from global best
                new_x, new_f = self.local_search(g_best, func, local_budget)
            else:  # otherwise, do local search on elite individuals
                new_x, new_f = self.local_search(elite_population[i], func, local_budget)
            if new_f < elite_fitness[i]:
                elite_fitness[i] = new_f
                elite_population[i] = new_x

        best_idx = np.argmin(elite_fitness)
        self.f_opt = elite_fitness[best_idx]
        self.x_opt = elite_population[best_idx]

        return self.f_opt, self.x_opt
