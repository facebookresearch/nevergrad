import numpy as np
import gym

class Booth:
    def __init__(self, dims=1):
        self.dims    = dims
        self.lb      = -10 * np.ones(dims)
        self.ub      =  10 * np.ones(dims)
        self.counter = 0

    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        result = ( x[0] + 2*x[1] - 7 )**2 + ( 2*x[0] + x[1] - 5 )**2
        result = result*-1
        return result

class Square:
    def __init__(self, dims=1):
        self.dims   = dims
        self.lb    = -10 * np.ones(dims)
        self.ub    =  10 * np.ones(dims)
        self.counter = 0

    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        result = np.inner( x, x )
        result = result
        return result

class Ackley:
    def __init__(self, dims=3):
        self.dims   = dims
        self.lb    = -15 * np.ones(dims)
        self.ub    =  30 * np.ones(dims)
        self.counter = 0

    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        w = 1 + (x - 1.0) / 4.0
        result = -1*(-20*np.exp(-0.2 * np.sqrt(np.inner(x,x) / x.size )) -np.exp(np.cos(2*np.pi*x).sum() /x.size) + 20 +np.e )
        result = result
        return result
        
class Rosenrock:
    def __init__(self, dims=3):
        self.dims    = dims
        self.lb      = -10  * np.ones(dims)
        self.ub      =  10 * np.ones(dims)
        self.counter = 0
        print("initialize rosenbrock at dims:", self.dims)
        
    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        
        result = sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
        result = -1*result
        
        return result
        
class Levy:
    def __init__(self, dims=3):
        self.dims    = dims
        self.lb      = -10 * np.ones(dims)
        self.ub      =  10 * np.ones(dims)
        self.counter = 0
        print("initialize rosenbrock at dims:", self.dims)
        
    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        
        w = []
        for idx in range(0, len(x)):
            w.append( 1 + (x[idx] - 1) / 4 )
        w = np.array(w)
        
        
        term1 = ( np.sin( np.pi*w[0] ) )**2;
        
        term3 = ( w[-1] - 1 )**2 * ( 1 + ( np.sin( 2 * np.pi * w[-1] ) )**2 );
        
        
        term2 = 0;
        for idx in range(1, len(w) ):
            wi  = w[idx]
            new = (wi-1)**2 * ( 1 + 10 * ( np.sin( np.pi* wi + 1 ) )**2)
            term2 = term2 + new
        
        result = term1 + term2 + term3
        result = result * -1
        return result
    
class Rastrigin:
    def __init__(self, dims=1):
        self.dims    = dims
        self.lb      = -5.12 * np.ones(dims)
        self.ub      =  5.12 * np.ones(dims)
        self.counter = 0

    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        
        tmp = 0;
        for idx in range(0, len(x)):
        	curt = x[idx];
        	tmp = tmp + (curt**2 - 10 * np.cos( 2 * np.pi * curt ) )

        result = 10 * len(x) + tmp
        result = result * -1
        return result
        
class Schwefel:
    def __init__(self, dims=1):
        self.dims    = dims
        self.lb      = -10 * np.ones(dims)
        self.ub      =  10 * np.ones(dims)
        self.counter = 0
    
    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        
        result = 0
        for idx in range(0, len(x)):
            curt = x[idx]
            result = result + curt*np.sin( np.sqrt( np.abs( curt ) ) )
        result = 418.9829*len(x) - result
        result = result *-1
        return result
        
class Hart6:
    def __init__(self):
        self.dims    = 6
        self.lb      = np.zeros(6)
        self.ub      = np.ones(6)
        self.counter = 0
    
    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        
        alpha = np.array( [1.0, 1.2, 3.0, 3.2] )

        A =  np.array( [ 
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14] ] )

        P = np.array( [[1312, 1696, 5569, 124, 8283, 5886],
                       [2329, 4135, 8307, 3736, 1004, 9991],
                       [2348, 1451, 3522, 2883, 3047, 6650],
                       [4047, 8828, 8732, 5743, 1091, 381]]) * 0.0001
                       
        outer = 0 
        for i in range(0, 4):
        	inner = 0
        	for j in range(0, 6):
        		xj    = x[j]
        		Aij   = A[i, j]
        		Pij   = P[i, j]
        		inner = inner + Aij*((xj-Pij)**2)
        	new   = alpha[i] * np.exp(-1*inner)
        	outer = outer + new
            
        y = -(2.58 + outer) / 1.94
        return y
        
class Lunarlanding:
    def __init__(self):
        self.dims = 12
        self.lb   = np.zeros(12)
        self.ub   = 2 * np.ones(12)
        self.counter = 0
        self.env = gym.make('LunarLander-v2')
        
    def heuristic_Controller(self, s, w):
        angle_targ = s[0] * w[0] + s[2] * w[1]
        if angle_targ > w[2]:
            angle_targ = w[2]
        if angle_targ < -w[2]:
            angle_targ = -w[2]
        hover_targ = w[3] * np.abs(s[0])

        angle_todo = (angle_targ - s[4]) * w[4] - (s[5]) * w[5]
        hover_todo = (hover_targ - s[1]) * w[6] - (s[3]) * w[7]

        if s[6] or s[7]:
            angle_todo = w[8]
            hover_todo = -(s[3]) * w[9]

        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > w[10]:
            a = 2
        elif angle_todo < -w[11]:
            a = 3
        elif angle_todo > +w[11]:
            a = 1
        return a
        
    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
    
        state = self.env.reset()
        rewards_for_episode = []
        num_steps = 2000
        
        for step in range(num_steps):
            # env.render()
            received_action = self.heuristic_Controller(state, x)
            next_state, reward, done, info = self.env.step(received_action)
            rewards_for_episode.append( reward )
            state = next_state
            if done:
                break
        rewards_for_episode = np.array(rewards_for_episode)
        
        return np.mean(rewards_for_episode)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
