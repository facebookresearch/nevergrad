import numpy as np
from scipy.optimize import minimize


class MultiSwarmAdaptiveDE_PSO:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.bounds = (-5.0, 5.0)
        self.swarm_size = 20
        self.num_swarms = 5

    def random_bounds(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], self.dim)

    def local_search(self, x, func):
        result = minimize(func, x, method="L-BFGS-B", bounds=[self.bounds] * self.dim)
        return result.x, result.fun

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Initialize multiple swarms
        swarms = [
            np.array([self.random_bounds() for _ in range(self.swarm_size)]) for _ in range(self.num_swarms)
        ]
        fitness = [np.array([func(ind) for ind in swarm]) for swarm in swarms]
        evaluations = self.swarm_size * self.num_swarms

        # PSO parameters
        w = 0.5  # Inertia weight
        c1 = 1.5  # Cognitive coefficient
        c2 = 1.5  # Social coefficient
        velocities = [np.random.uniform(-1, 1, (self.swarm_size, self.dim)) for _ in range(self.num_swarms)]
        local_bests = [swarm[np.argmin(fit)] for swarm, fit in zip(swarms, fitness)]
        local_best_fits = [min(fit) for fit in fitness]

        while evaluations < self.budget:
            new_swarms = []
            new_fitness = []
            best_swarm_idx = np.argmin(local_best_fits)

            for s in range(self.num_swarms):
                new_swarm = []
                new_fit = []

                for i in range(len(swarms[s])):
                    # PSO update
                    r1, r2 = np.random.rand(), np.random.rand()
                    velocities[s][i] = (
                        w * velocities[s][i]
                        + c1 * r1 * (local_bests[s] - swarms[s][i])
                        + c2
                        * r2
                        * (swarms[best_swarm_idx][np.argmin(fitness[best_swarm_idx])] - swarms[s][i])
                    )

                    trial_pso = swarms[s][i] + velocities[s][i]
                    trial_pso = np.clip(trial_pso, self.bounds[0], self.bounds[1])

                    # Mutation strategy from DE
                    F = 0.8
                    CR = 0.9
                    indices = np.arange(len(swarms[s]))
                    indices = np.delete(indices, i)
                    a, b, c = swarms[s][np.random.choice(indices, 3, replace=False)]
                    mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])

                    # Crossover
                    cross_points = np.random.rand(self.dim) < CR
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    trial = np.where(cross_points, mutant, trial_pso)

                    # Local Search
                    if np.random.rand() < 0.25 and evaluations < self.budget:
                        trial, f_trial = self.local_search(trial, func)
                        evaluations += 1
                    else:
                        f_trial = func(trial)
                        evaluations += 1

                    # Selection
                    if f_trial < fitness[s][i]:
                        new_swarm.append(trial)
                        new_fit.append(f_trial)
                        if f_trial < local_best_fits[s]:
                            local_best_fits[s] = f_trial
                            local_bests[s] = trial
                            if f_trial < self.f_opt:
                                self.f_opt = f_trial
                                self.x_opt = trial
                    else:
                        new_swarm.append(swarms[s][i])
                        new_fit.append(fitness[s][i])

                    if evaluations >= self.budget:
                        break

                new_swarms.append(np.array(new_swarm))
                new_fitness.append(np.array(new_fit))

                # Update local bests for the swarm
                local_bests[s] = new_swarms[s][np.argmin(new_fitness[s])]
                local_best_fits[s] = min(new_fitness[s])

            # Update swarms and fitness
            swarms = new_swarms
            fitness = new_fitness

            # Diversity Maintenance: Re-initialize if the population converges too tightly
            for s in range(self.num_swarms):
                if np.std(fitness[s]) < 1e-5 and evaluations < self.budget:
                    swarms[s] = np.array([self.random_bounds() for _ in range(self.swarm_size)])
                    fitness[s] = np.array([func(ind) for ind in swarms[s]])
                    evaluations += self.swarm_size

            # Adaptive parameter adjustment
            if np.random.rand() < 0.1:
                w = np.random.uniform(0.4, 0.9)
                c1 = np.random.uniform(1.0, 2.0)
                c2 = np.random.uniform(1.0, 2.0)

        return self.f_opt, self.x_opt
