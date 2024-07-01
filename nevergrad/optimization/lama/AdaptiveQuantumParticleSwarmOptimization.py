import numpy as np


class AdaptiveQuantumParticleSwarmOptimization:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initialize parameters
        self.f_opt = np.inf
        self.x_opt = None

        # Initialize swarm
        swarm_size = 20
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (swarm_size, self.dim))
        velocities = np.zeros_like(positions)
        personal_bests = positions.copy()
        personal_best_scores = np.array([np.inf] * swarm_size)

        # Global best
        global_best_position = None
        global_best_score = np.inf

        # Particle Swarm Optimization constants
        c1 = 1.5  # Cognitive constant
        c2 = 1.5  # Social constant
        w = 0.7  # Inertia weight

        # Quantum tunable parameters
        delta = 0.05  # Step size for quantum movement

        # Differential Evolution parameters
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover probability

        # Learning rate adaptation parameters
        alpha = 0.1  # Initial learning rate
        beta = 0.9  # Momentum term
        epsilon = 1e-8  # Small term to avoid division by zero

        prev_f = np.inf

        def quantum_move(x, g_best):
            return x + delta * (np.random.random(self.dim) * 2 - 1) * np.abs(g_best - x)

        # Hybrid loop (combining PSO, Gradient-based search, Differential Evolution, Quantum Movement, and Local Search)
        for i in range(self.budget):
            for idx in range(swarm_size):
                x = positions[idx]
                v = velocities[idx]

                # Evaluate the function at the current point
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

                # Update velocity and position using PSO
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                cognitive_component = c1 * r1 * (personal_bests[idx] - x)
                social_component = c2 * r2 * (global_best_position - x)
                velocities[idx] = w * v + cognitive_component + social_component
                positions[idx] = x + velocities[idx]
                positions[idx] = np.clip(positions[idx], self.lower_bound, self.upper_bound)

                # Gradient-based update
                grad = np.zeros_like(x)
                perturbation = 1e-5
                for j in range(self.dim):
                    x_perturb = x.copy()
                    x_perturb[j] += perturbation
                    grad[j] = (func(x_perturb) - f) / perturbation

                # Update the velocity and position using gradient
                velocity = beta * v - alpha * grad
                positions[idx] = x + velocity
                positions[idx] = np.clip(positions[idx], self.lower_bound, self.upper_bound)

                # Apply Differential Evolution mutation and crossover
                if np.random.rand() < CR:
                    indices = list(range(swarm_size))
                    indices.remove(idx)
                    a, b, c = np.random.choice(indices, 3, replace=False)
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

                # Apply quantum movement
                if np.random.rand() < 0.05:
                    quantum_position = quantum_move(x, global_best_position)
                    quantum_f = func(quantum_position)

                    if quantum_f < f:
                        positions[idx] = quantum_position
                        f = quantum_f

                        if quantum_f < personal_best_scores[idx]:
                            personal_best_scores[idx] = quantum_f
                            personal_bests[idx] = quantum_position.copy()

                        if quantum_f < global_best_score:
                            global_best_score = quantum_f
                            global_best_position = quantum_position.copy()

                        if quantum_f < self.f_opt:
                            self.f_opt = quantum_f
                            self.x_opt = quantum_position.copy()

                # Local Search for fine-tuning solutions
                if i % 10 == 0:  # Perform local search every 10 iterations
                    for _ in range(5):  # Number of local search steps
                        x_ls = x + np.random.normal(0, 0.1, self.dim)
                        x_ls = np.clip(x_ls, self.lower_bound, self.upper_bound)
                        f_ls = func(x_ls)

                        if f_ls < f:
                            positions[idx] = x_ls
                            f = f_ls

                            if f_ls < personal_best_scores[idx]:
                                personal_best_scores[idx] = f_ls
                                personal_bests[idx] = x_ls.copy()

                            if f_ls < global_best_score:
                                global_best_score = f_ls
                                global_best_position = x_ls.copy()

                            if f_ls < self.f_opt:
                                self.f_opt = f_ls
                                self.x_opt = x_ls.copy()

                # Adapt the learning rate based on the improvement
                if i > 0 and (prev_f - f) / (abs(prev_f) + epsilon) > 0.01:
                    alpha *= 1.05  # Increase learning rate if improvement is significant
                else:
                    alpha *= 0.7  # Decrease learning rate if improvement is not significant

                prev_f = f

        return self.f_opt, self.x_opt


# Usage example:
# optimizer = AdaptiveQuantumParticleSwarmOptimization(budget=10000)
# best_value, best_solution = optimizer(some_black_box_function)
