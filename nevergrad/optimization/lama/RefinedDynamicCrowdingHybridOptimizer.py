import numpy as np
from scipy.optimize import minimize


class RefinedDynamicCrowdingHybridOptimizer:
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
        crowding_factor=0.5,
        restart_threshold=1e-5,
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
        self.crowding_factor = crowding_factor
        self.dim = 5
        self.bounds = (-5.0, 5.0)
        self.eval_count = 0
        self.memory = []
        self.restart_threshold = restart_threshold

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

    def crowding_select(self, population, trial, fitness, f_trial):
        distances = np.linalg.norm(population - trial, axis=1)
        idx = np.argmin(distances)
        if f_trial < fitness[idx]:
            return idx
        else:
            return None

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
        local_search_budget_per_individual = local_search_budget // self.init_pop_size

        current_pop_size = self.init_pop_size
        successful_steps = []
        last_best_fitness = g_best_fitness

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
                r = np.random.choice(3)
                if r == 0:
                    a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])
                elif r == 1:
                    a, b = population[np.random.choice(idxs, 2, replace=False)]
                    mutant = np.clip(a + F * (b - population[i]), self.bounds[0], self.bounds[1])
                else:
                    a = population[np.random.choice(idxs)]
                    mutant = np.clip(a + F * (g_best - population[i]), self.bounds[0], self.bounds[1])

                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Incorporate additional crossover mechanism with adaptive blending
                if np.random.rand() < 0.3:  # 30% chance to apply blend crossover
                    partner_idx = np.random.choice(idxs)
                    partner = population[partner_idx]
                    trial = 0.5 * (trial + partner)
                    trial = np.clip(trial, self.bounds[0], self.bounds[1])

                # Selection with crowding mechanism
                f_trial = func(trial)
                self.eval_count += 1
                selected_idx = self.crowding_select(population, trial, fitness, f_trial)
                if selected_idx is not None and selected_idx < current_pop_size:
                    fitness[selected_idx] = f_trial
                    population[selected_idx] = trial
                    successful_steps.append((F, CR))
                    if len(successful_steps) > 50:
                        successful_steps.pop(0)
                    # Self-adapting parameters
                    F_values[selected_idx] = min(F * 1.1, 1.0)
                    CR_values[selected_idx] = min(CR * 1.1, 1.0)

                    # Update personal best
                    if f_trial < p_best_fitness[selected_idx]:
                        p_best[selected_idx] = trial
                        p_best_fitness[selected_idx] = f_trial

                    # Update global best
                    if f_trial < g_best_fitness:
                        g_best = trial
                        g_best_fitness = f_trial

                if self.eval_count >= global_search_budget:
                    break

            # Dynamic population resizing
            if self.eval_count < global_search_budget / 2:
                current_pop_size = int(self.init_pop_size * (1 - self.eval_count / global_search_budget))
                current_pop_size = max(current_pop_size, self.min_pop_size)
                population = population[:current_pop_size]
                fitness = fitness[:current_pop_size]
                velocities = velocities[:current_pop_size]
                F_values = F_values[:current_pop_size]
                CR_values = CR_values[:current_pop_size]
                p_best = p_best[:current_pop_size]
                p_best_fitness = p_best_fitness[:current_pop_size]

            # Restart mechanism based on convergence criterion
            if abs(last_best_fitness - g_best_fitness) < self.restart_threshold:
                population = np.random.uniform(self.bounds[0], self.bounds[1], (self.init_pop_size, self.dim))
                fitness = np.array([func(ind) for ind in population])
                self.eval_count += self.init_pop_size
                velocities = np.random.uniform(-1, 1, (self.init_pop_size, self.dim))
                p_best = population.copy()
                p_best_fitness = fitness.copy()
                g_best = population[np.argmin(fitness)]
                g_best_fitness = np.min(fitness)
                successful_steps = []
            last_best_fitness = g_best_fitness

        # Perform local search on the best individuals
        for i in range(current_pop_size):
            if self.eval_count >= self.budget:
                break
            local_budget = min(local_search_budget_per_individual, self.budget - self.eval_count)
            new_x, new_f = self.local_search(population[i], func, local_budget)
            if new_f < fitness[i]:
                fitness[i] = new_f
                population[i] = new_x

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        return self.f_opt, self.x_opt
