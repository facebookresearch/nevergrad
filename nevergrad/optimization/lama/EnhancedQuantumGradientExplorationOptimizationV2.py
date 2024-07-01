import numpy as np


class EnhancedQuantumGradientExplorationOptimizationV2:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        swarm_size = 20
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (swarm_size, self.dim))
        velocities = np.zeros_like(positions)
        personal_bests = positions.copy()
        personal_best_scores = np.array([np.inf] * swarm_size)

        global_best_position = None
        global_best_score = np.inf

        c1, c2 = 2.0, 2.0
        w_max, w_min = 0.9, 0.4

        alpha = 0.1
        beta = 0.9
        epsilon = 1e-8

        F_min, F_max = 0.4, 0.9
        CR = 0.9

        diversity_threshold = 0.1
        stagnation_counter = 0
        max_stagnation = 20

        theta = np.pi / 4
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        mutation_factor = 0.2

        improvement_threshold = 0.005

        historical_bests = []

        prev_f = np.inf

        for i in range(self.budget):
            w = w_max - (w_max - w_min) * (i / self.budget)
            T = 1 - (i / self.budget)

            for idx in range(swarm_size):
                x, v = positions[idx], velocities[idx]

                f = func(x)
                if f < personal_best_scores[idx]:
                    personal_best_scores[idx] = f
                    personal_bests[idx] = x.copy()

                if f < global_best_score:
                    global_best_score = f
                    global_best_position = x.copy()

                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = x.copy()

                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                cog_comp = c1 * r1 * (personal_bests[idx] - x)
                soc_comp = c2 * r2 * (global_best_position - x)
                velocities[idx] = w * v + cog_comp + soc_comp
                positions[idx] = x + velocities[idx]
                positions[idx] = np.clip(positions[idx], self.lower_bound, self.upper_bound)

                grad = np.zeros_like(x)
                perturbation = 1e-5
                for j in range(self.dim):
                    x_perturb = x.copy()
                    x_perturb[j] += perturbation
                    grad[j] = (func(x_perturb) - f) / perturbation

                velocity = beta * v - alpha * grad
                positions[idx] = x + velocity
                positions[idx] = np.clip(positions[idx], self.lower_bound, self.upper_bound)

                if np.random.rand() < CR:
                    indices = list(range(swarm_size))
                    indices.remove(idx)
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    F = F_min + (F_max - F_min) * np.random.rand()
                    mutant = personal_bests[a] + F * (personal_bests[b] - personal_bests[c])
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                    trial = np.where(np.random.rand(self.dim) < CR, mutant, x)
                    trial_f = func(trial)

                    if trial_f < f:
                        positions[idx] = trial
                        f = trial_f

                        if trial_f < personal_best_scores[idx]:
                            personal_best_scores[idx] = trial_f
                            personal_bests[idx] = trial.copy()

                        if trial_f < global_best_score:
                            global_best_score = trial_f
                            global_best_position = trial.copy()

                        if trial_f < self.f_opt:
                            self.f_opt = trial_f
                            self.x_opt = trial.copy()

                if i > 0 and (prev_f - f) / abs(prev_f) > improvement_threshold:
                    alpha *= 1.1
                else:
                    alpha *= 0.9

                prev_f = f

            if stagnation_counter >= max_stagnation:
                for idx in range(swarm_size):
                    if np.linalg.norm(positions[idx] - global_best_position) < diversity_threshold:
                        positions[idx] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                stagnation_counter = 0

            if i > 0 and prev_f == self.f_opt:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            if len(historical_bests) > 0 and i % 50 == 0:
                for idx in range(swarm_size):
                    new_position = historical_bests[
                        np.random.randint(len(historical_bests))
                    ] + mutation_factor * np.random.uniform(-1, 1, self.dim)
                    new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
                    new_f = func(new_position)
                    if new_f < personal_best_scores[idx]:
                        personal_best_scores[idx] = new_f
                        personal_bests[idx] = new_position
                        positions[idx] = new_position
                    if new_f < global_best_score:
                        global_best_score = new_f
                        global_best_position = new_position
                    if new_f < self.f_opt:
                        self.f_opt = new_f
                        self.x_opt = new_position

            if i % 100 == 0 and i > 0:
                for idx in range(swarm_size):
                    new_position = np.dot(rotation_matrix, positions[idx])
                    new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
                    new_f = func(new_position)
                    if new_f < personal_best_scores[idx]:
                        personal_best_scores[idx] = new_f
                        personal_bests[idx] = new_position
                        positions[idx] = new_position
                    if new_f < global_best_score:
                        global_best_score = new_f
                        global_best_position = new_position
                    if new_f < self.f_opt:
                        self.f_opt = new_f
                        self.x_opt = new_position

            if i % 200 == 0 and i > 0:
                for idx in range(swarm_size):
                    mutation = mutation_factor * np.random.uniform(-1, 1, self.dim)
                    new_position = positions[idx] + mutation
                    new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
                    new_f = func(new_position)
                    if new_f < personal_best_scores[idx]:
                        personal_best_scores[idx] = new_f
                        personal_bests[idx] = new_position
                        positions[idx] = new_position
                    if new_f < global_best_score:
                        global_best_score = new_f
                        global_best_position = new_position
                    if new_f < self.f_opt:
                        self.f_opt = new_f
                        self.x_opt = new_position

            prev_f = self.f_opt

            historical_bests.append(global_best_position)
            if len(historical_bests) > 10:
                historical_bests.pop(0)

        return self.f_opt, self.x_opt


# Usage example:
# optimizer = EnhancedQuantumGradientExplorationOptimizationV2(budget=10000)
# best_value, best_solution = optimizer(some_black_box_function)
