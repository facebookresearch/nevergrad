import numpy as np
from scipy.optimize import minimize


class DynamicNichePSO_DE_LS:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.bounds = (-5.0, 5.0)
        self.swarm_size = 20
        self.init_num_niches = 5
        self.alpha = 0.5  # Weight for DE contribution
        self.beta = 0.5  # Weight for PSO contribution

    def random_bounds(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], self.dim)

    def local_search(self, x, func):
        result = minimize(func, x, method="L-BFGS-B", bounds=[(self.bounds[0], self.bounds[1])] * self.dim)
        return result.x, result.fun

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Initialize niches
        niches = [
            np.array([self.random_bounds() for _ in range(self.swarm_size)])
            for _ in range(self.init_num_niches)
        ]
        fitness = [np.array([func(ind) for ind in niche]) for niche in niches]
        evaluations = self.swarm_size * self.init_num_niches

        # PSO parameters
        w = 0.5  # Inertia weight
        c1 = 1.5  # Cognitive coefficient
        c2 = 1.5  # Social coefficient
        velocities = [
            np.random.uniform(-1, 1, (self.swarm_size, self.dim)) for _ in range(self.init_num_niches)
        ]
        local_bests = [niches[i][np.argmin(fitness[i])] for i in range(len(niches))]
        local_best_fits = [min(fitness[i]) for i in range(len(niches))]
        global_best = local_bests[np.argmin(local_best_fits)]
        global_best_fit = min(local_best_fits)

        while evaluations < self.budget:
            new_niches = []
            new_fitness = []

            for n in range(len(niches)):
                new_niche = []
                new_fit = []

                for i in range(len(niches[n])):
                    # PSO update
                    r1, r2 = np.random.rand(), np.random.rand()
                    velocities[n][i] = (
                        w * velocities[n][i]
                        + c1 * r1 * (local_bests[n] - niches[n][i])
                        + c2 * r2 * (global_best - niches[n][i])
                    )

                    trial_pso = niches[n][i] + velocities[n][i]
                    trial_pso = np.clip(trial_pso, self.bounds[0], self.bounds[1])

                    # Mutation strategy from DE
                    F = 0.8
                    CR = 0.9
                    indices = np.arange(len(niches[n]))
                    indices = np.delete(indices, i)
                    a, b, c = niches[n][np.random.choice(indices, 3, replace=False)]
                    mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])

                    # Crossover
                    cross_points = np.random.rand(self.dim) < CR
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    trial_de = np.where(cross_points, mutant, niches[n][i])

                    # Combined DE and PSO trial
                    trial = self.alpha * trial_de + self.beta * trial_pso
                    trial = np.clip(trial, self.bounds[0], self.bounds[1])

                    # Local Search
                    if np.random.rand() < 0.3 and evaluations < self.budget:
                        trial, f_trial = self.local_search(trial, func)
                        evaluations += 1
                    else:
                        f_trial = func(trial)
                        evaluations += 1

                    # Selection
                    if f_trial < fitness[n][i]:
                        new_niche.append(trial)
                        new_fit.append(f_trial)
                        if f_trial < local_best_fits[n]:
                            local_best_fits[n] = f_trial
                            local_bests[n] = trial
                            if f_trial < global_best_fit:
                                global_best_fit = f_trial
                                global_best = trial
                                if f_trial < self.f_opt:
                                    self.f_opt = f_trial
                                    self.x_opt = trial
                    else:
                        new_niche.append(niches[n][i])
                        new_fit.append(fitness[n][i])

                    if evaluations >= self.budget:
                        break

                new_niches.append(np.array(new_niche))
                new_fitness.append(np.array(new_fit))

            # Update niches and fitness
            niches = new_niches
            fitness = new_fitness

            # Diversity Maintenance: Re-initialize if the population converges too tightly
            for n in range(len(niches)):
                if np.std(fitness[n]) < 1e-5 and evaluations < self.budget:
                    niches[n] = np.array([self.random_bounds() for _ in range(self.swarm_size)])
                    fitness[n] = np.array([func(ind) for ind in niches[n]])
                    evaluations += self.swarm_size

            # Dynamic niching and regrouping
            if evaluations % (self.swarm_size * self.init_num_niches) == 0:
                all_particles = np.concatenate(niches)
                all_fitness = np.concatenate(fitness)
                sorted_indices = np.argsort(all_fitness)
                num_niches = max(2, len(niches) // 2)
                niches = [all_particles[sorted_indices[i::num_niches]] for i in range(num_niches)]
                fitness = [all_fitness[sorted_indices[i::num_niches]] for i in range(num_niches)]
                velocities = [np.random.uniform(-1, 1, (len(niche), self.dim)) for niche in niches]
                local_bests = [niches[i][0] for i in range(num_niches)]
                local_best_fits = [fitness[i][0] for i in range(num_niches)]

            # Adaptive parameter adjustment
            w = np.random.uniform(0.4, 0.9)
            c1 = np.random.uniform(1.0, 2.0)
            c2 = np.random.uniform(1.0, 2.0)

            # Adding a restart mechanism based on diversity and performance
            if evaluations % (self.swarm_size * self.init_num_niches * 10) == 0:
                diversity = np.mean([np.std(fit) for fit in fitness])
                if diversity < 1e-3:
                    niches = [
                        np.array([self.random_bounds() for _ in range(self.swarm_size)])
                        for _ in range(self.init_num_niches)
                    ]
                    fitness = [np.array([func(ind) for ind in niches[n]]) for n in range(len(niches))]
                    evaluations += self.swarm_size * self.init_num_niches

        return self.f_opt, self.x_opt
