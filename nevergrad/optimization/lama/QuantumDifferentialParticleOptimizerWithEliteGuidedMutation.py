import numpy as np
from scipy.optimize import minimize


class QuantumDifferentialParticleOptimizerWithEliteGuidedMutation:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.bounds = (-5.0, 5.0)
        self.swarm_size = 60
        self.num_elites = 5
        self.alpha = 0.6
        self.beta = 0.4
        self.local_search_prob = 0.4

    def random_bounds(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], self.dim)

    def local_search(self, x, func):
        result = minimize(func, x, method="L-BFGS-B", bounds=[(self.bounds[0], self.bounds[1])] * self.dim)
        return result.x, result.fun

    def quantum_update(self, x, elites, beta):
        p_best = elites[np.random.randint(len(elites))]
        u = np.random.uniform(0, 1, self.dim)
        v = np.random.uniform(-1, 1, self.dim)
        Q = beta * (p_best - x) * np.log(1 / u)
        return x + Q * v

    def enhanced_adaptive_restart(
        self, particles, fitness, personal_bests, personal_best_fits, global_best, global_best_fit, func
    ):
        std_dev = np.std(personal_best_fits)
        mean_fit = np.mean(personal_best_fits)

        if std_dev < 1e-3 or mean_fit < global_best_fit * 1.01:
            particles = np.array([self.random_bounds() for _ in range(self.swarm_size)])
            fitness = np.array([func(ind) for ind in particles])

            personal_bests = np.copy(particles)
            personal_best_fits = np.copy(fitness)

            global_best = particles[np.argmin(fitness)]
            global_best_fit = np.min(fitness)
        else:
            global_best = global_best
            global_best_fit = global_best_fit

        return particles, fitness, personal_bests, personal_best_fits, global_best, global_best_fit

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        particles = np.array([self.random_bounds() for _ in range(self.swarm_size)])
        fitness = np.array([func(ind) for ind in particles])
        evaluations = self.swarm_size

        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_bests = np.copy(particles)
        personal_best_fits = np.copy(fitness)
        global_best = particles[np.argmin(fitness)]
        global_best_fit = np.min(fitness)

        w = 0.5
        c1 = 1.5
        c2 = 1.5

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (
                    w * velocities[i]
                    + c1 * r1 * (personal_bests[i] - particles[i])
                    + c2 * r2 * (global_best - particles[i])
                )

                trial_pso = particles[i] + velocities[i]
                trial_pso = np.clip(trial_pso, self.bounds[0], self.bounds[1])

                F = 0.8
                CR = 0.9
                indices = np.arange(self.swarm_size)
                indices = np.delete(indices, i)
                a, b, c = particles[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])

                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial_de = np.where(cross_points, mutant, particles[i])

                if np.random.rand() < self.local_search_prob and evaluations < self.budget:
                    trial, f_trial = self.local_search(trial_de, func)
                    evaluations += 1
                else:
                    f_trial = func(trial_de)
                    evaluations += 1

                elite_particles = personal_bests[np.argsort(personal_best_fits)[: self.num_elites]]
                trial = self.quantum_update(trial_de, elite_particles, self.beta)
                trial = np.clip(trial, self.bounds[0], self.bounds[1])
                f_trial = func(trial)
                evaluations += 1

                if f_trial < fitness[i]:
                    particles[i] = trial
                    fitness[i] = f_trial
                    if f_trial < personal_best_fits[i]:
                        personal_bests[i] = trial
                        personal_best_fits[i] = f_trial
                        if f_trial < global_best_fit:
                            global_best_fit = f_trial
                            global_best = trial
                            if f_trial < self.f_opt:
                                self.f_opt = f_trial
                                self.x_opt = trial

            w = np.random.uniform(0.3, 0.9)
            c1 = np.random.uniform(1.0, 2.5)
            c2 = np.random.uniform(1.0, 2.5)

            if evaluations % (self.swarm_size * 2) == 0:
                improvement = (self.f_opt - global_best_fit) / self.f_opt if self.f_opt != 0 else 0
                if improvement < 0.01:
                    self.local_search_prob = min(1.0, self.local_search_prob + 0.1)
                else:
                    self.local_search_prob = max(0.1, self.local_search_prob - 0.1)

            (
                particles,
                fitness,
                personal_bests,
                personal_best_fits,
                global_best,
                global_best_fit,
            ) = self.enhanced_adaptive_restart(
                particles, fitness, personal_bests, personal_best_fits, global_best, global_best_fit, func
            )

            if evaluations % (self.swarm_size * 10) == 0:
                diversity = np.std(fitness)
                if diversity < 1e-3:
                    # Refinement step for elite particles
                    for j in range(self.num_elites):
                        elite, elite_fit = self.local_search(personal_bests[j], func)
                        evaluations += 1
                        if elite_fit < personal_best_fits[j]:
                            personal_bests[j] = elite
                            personal_best_fits[j] = elite_fit
                            if elite_fit < global_best_fit:
                                global_best_fit = elite_fit
                                global_best = elite
                                if elite_fit < self.f_opt:
                                    self.f_opt = elite_fit
                                    self.x_opt = elite

                    particles = np.array([self.random_bounds() for _ in range(self.swarm_size)])
                    fitness = np.array([func(ind) for ind in particles])
                    evaluations += self.swarm_size

        return self.f_opt, self.x_opt
