import numpy as np


class QuantumGradientHybridOptimization:
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
        swarm_size = 25  # Increased swarm size to improve exploration
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (swarm_size, self.dim))
        velocities = np.zeros_like(positions)
        personal_bests = positions.copy()
        personal_best_scores = np.array([np.inf] * swarm_size)

        # Global best
        global_best_position = None
        global_best_score = np.inf

        # Particle Swarm Optimization constants
        c1 = 1.8  # Cognitive constant (balanced for more exploration)
        c2 = 1.8  # Social constant (balanced for more exploration)
        w = 0.6  # Inertia weight (balanced to control convergence)

        # Learning rate adaptation parameters
        alpha = 0.05  # Initial learning rate (reduced for finer adjustments)
        beta = 0.9  # Momentum term
        epsilon = 1e-8  # Small term to avoid division by zero

        # Differential Evolution parameters
        F_min = 0.4  # Lower bound for differential weight
        F_max = 0.9  # Upper bound for differential weight
        CR = 0.9  # Crossover probability

        # Diversity enforcement parameters
        diversity_threshold = 0.15
        stagnation_counter = 0
        max_stagnation = 15  # Reduced stagnation threshold to trigger diversity enforcement sooner

        # Exploration improvement parameters
        exploration_factor = 0.3  # Exploration factor to enhance exploration phase

        # Quantum-inspired rotation matrix
        theta = np.pi / 4  # Rotation angle
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        # Mutation factor for mutation-based exploration
        mutation_factor = 0.25

        # Adaptive threshold for learning rate tuning
        improvement_threshold = 0.01

        prev_f = np.inf

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

                # Adapt the learning rate based on the improvement
                if i > 0 and (prev_f - f) / abs(prev_f) > improvement_threshold:
                    alpha *= 1.1  # Increase learning rate if improvement is significant
                else:
                    alpha *= 0.9  # Decrease learning rate if improvement is not significant

                prev_f = f

            # Check for stagnation and enforce diversity if needed
            if stagnation_counter >= max_stagnation:
                for idx in range(swarm_size):
                    if np.linalg.norm(positions[idx] - global_best_position) < diversity_threshold:
                        positions[idx] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                stagnation_counter = 0

            if i > 0 and prev_f == self.f_opt:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            # Dynamic exploration phase
            if stagnation_counter >= max_stagnation:
                for idx in range(swarm_size):
                    new_position = global_best_position + exploration_factor * np.random.uniform(
                        -1, 1, self.dim
                    )
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

                stagnation_counter = 0

            # Quantum-inspired exploration using rotation matrix
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

            # Mutation-based exploration
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

        return self.f_opt, self.x_opt


# Usage example:
# optimizer = QuantumGradientHybridOptimization(budget=10000)
# best_value, best_solution = optimizer(some_black_box_function)
