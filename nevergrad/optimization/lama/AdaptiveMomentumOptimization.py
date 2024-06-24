import numpy as np


class AdaptiveMomentumOptimization:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0
        self.alpha = 0.1  # Learning rate
        self.beta1 = 0.9  # Momentum term
        self.beta2 = 0.999  # RMSProp term
        self.epsilon = 1e-8  # To prevent division by zero

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        x = np.random.uniform(self.lb, self.ub, self.dim)
        m = np.zeros(self.dim)
        v = np.zeros(self.dim)

        for t in range(1, self.budget + 1):
            # Evaluate function
            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x.copy()

            # Numerical gradient estimation
            grad = self._approx_gradient(func, x)

            # Update biased first moment estimate
            m = self.beta1 * m + (1 - self.beta1) * grad

            # Update biased second raw moment estimate
            v = self.beta2 * v + (1 - self.beta2) * (grad**2)

            # Compute bias-corrected first moment estimate
            m_hat = m / (1 - self.beta1**t)

            # Compute bias-corrected second raw moment estimate
            v_hat = v / (1 - self.beta2**t)

            # Update parameters
            x -= self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)

            # Ensure the solutions remain within bounds
            x = np.clip(x, self.lb, self.ub)

        return self.f_opt, self.x_opt

    def _approx_gradient(self, func, x):
        # Gradient approximation using central difference
        grad = np.zeros(self.dim)
        h = 1e-5  # Step size for numerical differentiation
        for i in range(self.dim):
            x_forward = x.copy()
            x_backward = x.copy()
            x_forward[i] += h
            x_backward[i] -= h

            f_forward = func(x_forward)
            f_backward = func(x_backward)

            grad[i] = (f_forward - f_backward) / (2 * h)

        return grad
