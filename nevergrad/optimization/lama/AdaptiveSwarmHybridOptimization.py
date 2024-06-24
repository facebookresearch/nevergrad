import numpy as np


class AdaptiveSwarmHybridOptimization:
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
        swarm_size = 25
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

        # Adaptive Learning Rate parameters
        alpha = 0.1  # Initial learning rate
        beta = 0.9  # Momentum term
        epsilon = 1e-8  # Small term to avoid division by zero

        # Differential Evolution parameters
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover probability

        # Hybrid loop (combining PSO, Gradient-based search, and Differential Evolution)
        fitness_history = []
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

                # Update the velocity and position using gradient descent
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

                # Adaptive learning rate strategy
                if i > 0 and len(fitness_history) > 0:
                    recent_improvement = np.mean(np.diff(fitness_history[-5:]))
                    if recent_improvement < 0:
                        alpha = min(alpha * 1.05, 1.0)  # Increase learning rate if recent improvement
                    else:
                        alpha = max(alpha * 0.7, 0.01)  # Decrease learning rate if no recent improvement

                fitness_history.append(f)
                prev_f = f

        return self.f_opt, self.x_opt
