import numpy as np


class HybridAdaptiveSearch:
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
        swarm_size = 10
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

        # Learning rate adaptation parameters
        alpha = 0.1  # Initial learning rate
        beta = 0.9  # Momentum term
        epsilon = 1e-8  # Small term to avoid division by zero

        # Hybrid loop (combining PSO and Gradient-based search)
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

                # Adapt the learning rate based on the improvement
                if i > 0 and (prev_f - f) / abs(prev_f) > 0.01:
                    alpha *= 1.05  # Increase learning rate if improvement is significant
                else:
                    alpha *= 0.7  # Decrease learning rate if improvement is not significant

                prev_f = f

        return self.f_opt, self.x_opt
