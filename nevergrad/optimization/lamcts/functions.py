#     # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#     # verify
#     # This source code is licensed under the MIT license found in the
#     # LICENSE file in the root directory of this source tree.
#     import numpy as np
#     import gym
#     import json
#     import os
#     from .push_function import PushReward
#     #from .rover_function import rover_nag



#     class tracker:
#         def __init__(self, foldername):
#             self.counter   = 0
#             self.results   = []
#             self.curt_best = float("inf")
#             self.foldername = foldername
#             try:
#                 os.mkdir(foldername)
#             except OSError:
#                 print ("Creation of the directory %s failed" % foldername)
#             else:
#                 print ("Successfully created the directory %s " % foldername)
#             
#         def dump_trace(self):
#             trace_path = self.foldername + '/result' + str(len( self.results) )
#             final_results_str = json.dumps(self.results)
#             with open(trace_path, "a") as f:
#                 f.write(final_results_str + '\n')
#                 
#         def track(self, result):
#             if result < self.curt_best:
#                 self.curt_best = result
#             self.results.append(self.curt_best)
#             if len(self.results) % 100 == 0:
#                 self.dump_trace()
#             

#     class Square:
#         def __init__(self, dims=1):
#             self.dims   = dims
#             self.lb    = -10 * np.ones(dims)
#             self.ub    =  10 * np.ones(dims)
#             self.counter = 0

#         def __call__(self, x):
#             self.counter += 1
#             assert len(x) == self.dims
#             assert x.ndim == 1
#             #assert np.all(x <= self.ub) and np.all(x >= self.lb)
#             print("=>counter", self.counter)
#             result = np.inner( x, x )
#             result = result
#             return result


#         
#     class Rastrigin:
#         def __init__(self, dims=1):
#             self.dims    = dims
#             self.lb      = -5.12 * np.ones(dims)
#             self.ub      =  5.12 * np.ones(dims)
#             self.counter = 0
#             self.iteration = 1000
#             self.tracker = tracker('Rastrigin')

#         def __call__(self, x):
#             self.counter += 1
#             assert len(x) == self.dims
#             assert x.ndim == 1
#             # #assert np.all(x <= self.ub) and np.all(x >= self.lb)

#             tmp = 0;
#             for idx in range(0, len(x)):
#                     curt = x[idx];
#                     tmp = tmp + (curt**2 - 10 * np.cos( 2 * np.pi * curt ) )

#             result = 10 * len(x) + tmp
#             self.tracker.track( result )
#             if self.counter > self.iteration:
#                 os._exit(1)
#             
#             return result
#             
#     class Ackley:
#         def __init__(self, dims=3):
#             self.dims   = dims
#             self.lb    = -5 * np.ones(dims)
#             self.ub    =  10 * np.ones(dims)
#             self.counter = 0
#             if dims == 20:
#                 self.iteration = 1000
#             else:
#                 self.iteration = 10000
#             self.tracker = tracker('Ackley'+str(dims) )
#             

#         def __call__(self, x):
#             self.counter += 1
#             assert len(x) == self.dims
#             assert x.ndim == 1
#             # #assert np.all(x <= self.ub) and np.all(x >= self.lb)
#             w = 1 + (x - 1.0) / 4.0
#             result = (-20*np.exp(-0.2 * np.sqrt(np.inner(x,x) / x.size )) -np.exp(np.cos(2*np.pi*x).sum() /x.size) + 20 +np.e )
#             
#             self.tracker.track( result )
#             
#             if self.counter > self.iteration:
#                 os._exit(1)
#             
#             return result
#             
#     class Rosenrock:
#         def __init__(self, dims=3):
#             self.dims    = dims
#             self.lb      = -10 * np.ones(dims)
#             self.ub      =  10 * np.ones(dims)
#             self.counter =  0
#             self.tracker = tracker('Rosenrock'+str(dims) )
#             if dims  == 20:
#                 self.iteration = 1000
#             else:
#                 self.iteration = 10000
#             
#             print("initialize rosenbrock at dims:", self.dims)

#         def __call__(self, x):
#             self.counter += 1
#             assert len(x) == self.dims
#             assert x.ndim == 1
#             ##assert np.all(x <= self.ub) and np.all(x >= self.lb)

#             result = sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
#             
#             self.tracker.track( result )
#             
#             if self.counter > self.iteration:
#                 os._exit(1)

#             return result        

#             
#          
#     class Push:
#         def __init__(self):
#             self.PushReward = PushReward()
#             self.counter = 0
#             self.dims = self.PushReward.dims
#             self.lb   = self.PushReward.lb
#             self.ub   = self.PushReward.ub
#             self.tracker = tracker('push16d')
#             self.iteration = 10000
#             self.dims = self.PushReward.dims
#             
#         def __call__(self, x):
#             self.counter += 1
#             assert len(x) == self.PushReward.dims
#             assert x.ndim == 1
#             #assert np.all(x <= self.PushReward.ub) and np.all(x >= self.PushReward.lb)
#             reward = self.PushReward(x)
#             if self.counter % 1000 == 0:
#                 print(self.counter)
#             
#             self.tracker.track( reward )
#             
#             if self.counter > self.iteration:
#                 os._exit(1)
#                 
#             return reward
#             
#     class Rover:
#         def __init__(self):
#             self.RoverReward = rover_nag()
#             self.counter = 0
#             self.dims = self.RoverReward.dims
#             self.lb   = self.RoverReward.lb
#             self.ub   = self.RoverReward.ub
#             self.tracker = tracker('rover60d')
#             self.iteration = 20000
#             
#             
#         def __call__(self, x):
#             self.counter += 1
#             assert len(x) == self.RoverReward.dims
#             assert x.ndim == 1
#             #assert np.all(x <= self.RoverReward.ub) and np.all(x >= self.RoverReward.lb)
#             reward = self.RoverReward(x)
#             
#             self.tracker.track( reward )
#             
#             if self.counter > self.iteration:
#                 os._exit(1)
#             return reward
#         
#        
#             
#     class Lunarlanding:
#         def __init__(self):
#             self.dims = 12
#             self.lb   = np.zeros(12)
#             self.ub   = 2 * np.ones(12)
#             self.counter = 0
#             self.env = gym.make('LunarLander-v2')
#             self.tracker = tracker('Lunarlanding')
#             self.iteration = 2000

#             
#         def heuristic_Controller(self, s, w):
#             angle_targ = s[0] * w[0] + s[2] * w[1]
#             if angle_targ > w[2]:
#                 angle_targ = w[2]
#             if angle_targ < -w[2]:
#                 angle_targ = -w[2]
#             hover_targ = w[3] * np.abs(s[0])

#             angle_todo = (angle_targ - s[4]) * w[4] - (s[5]) * w[5]
#             hover_todo = (hover_targ - s[1]) * w[6] - (s[3]) * w[7]

#             if s[6] or s[7]:
#                 angle_todo = w[8]
#                 hover_todo = -(s[3]) * w[9]

#             a = 0
#             if hover_todo > np.abs(angle_todo) and hover_todo > w[10]:
#                 a = 2
#             elif angle_todo < -w[11]:
#                 a = 3
#             elif angle_todo > +w[11]:
#                 a = 1
#             return a
#             
#         def __call__(self, x):
#             self.counter += 1
#             assert len(x) == self.dims
#             assert x.ndim == 1
#             ##assert np.all(x <= self.ub) and np.all(x >= self.lb)
#         
#             total_rewards = []
#             for i in range(0, 5):
#                 state = self.env.reset()
#                 rewards_for_episode = []
#                 num_steps = 2000
#             
#                 for step in range(num_steps):
#                     # env.render()
#                     received_action = self.heuristic_Controller(state, x)
#                     next_state, reward, done, info = self.env.step(received_action)
#                     rewards_for_episode.append( reward )
#                     state = next_state
#                     if done:
#                         break
#                 rewards_for_episode = np.array(rewards_for_episode)
#                 total_rewards.append( np.sum(rewards_for_episode) )
#             final_result = np.mean(total_rewards)*-1
#             self.tracker.track( final_result )
#             
#             if self.counter > self.iteration:
#                 os._exit(1)
#             
#             return final_result
#             
#     class Ant:
#         
#         def __init__(self):
#             lin_policy   = np.load("./MujucoPolicies/Ant-v1/lin_policy_plus.npz", allow_pickle = True)
#             lin_policy   = lin_policy['arr_0']
#             self.policy  = lin_policy[0]
#             self.mean    = lin_policy[1]
#             self.std     = lin_policy[2]
#             self.dims    = len( self.policy.ravel() )
#             self.lb      = -1 * np.ones(self.dims)
#             self.ub      =  1 * np.ones(self.dims)
#             self.counter = 0
#             self.env          = gym.make('Ant-v2')
#             self.num_rollouts = 1
#             self.render  = False
#             self.tracker = tracker('Ant')
#             self.iteration = 40000

#             
#         
#         def __call__(self, x):
#             self.counter += 1
#             assert len(x) == self.dims
#             assert x.ndim == 1
#             ##assert np.all(x <= self.ub) and np.all(x >= self.lb)
#             
#             M = x.reshape(self.policy.shape)
#             
#             returns = []
#             observations = []
#             actions = []
#             
#             for i in range(self.num_rollouts):
#                 obs    = self.env.reset()
#                 done   = False
#                 totalr = 0.
#                 steps  = 0
#                 while not done:
#                     action = np.dot(M, (obs - self.mean)/self.std)
#                     observations.append(obs)
#                     actions.append(action)
#                     obs, r, done, _ = self.env.step(action)
#                     totalr += r
#                     steps += 1
#                     if self.render:
#                         self.env.render()
#                     # print(env.spec)
#                     # if steps % 100 == 0: print("%i/%i"%(steps, env.spec.timestep_limit))
#                     #if steps >= env.spec.timestep_limit:
#                     #    break
#                 returns.append(totalr)
#             final_result = np.mean(returns)*-1
#             self.tracker.track( final_result )
#             
#             if self.counter > self.iteration:
#                 os._exit(1)
#             
#             return final_result      
#             
#     class Swimmer:
#         
#         def __init__(self):
#             lin_policy   = np.load("./MujucoPolicies/Swimmer-v1/lin_policy_plus.npz", allow_pickle = True)
#             lin_policy   = lin_policy['arr_0']
#             self.policy  = lin_policy[0]
#             self.mean    = lin_policy[1]
#             self.std     = lin_policy[2]
#             self.dims    = len( self.policy.ravel() )
#             self.lb      = -1 * np.ones(self.dims)
#             self.ub      =  1 * np.ones(self.dims)
#             self.counter = 0
#             self.env          = gym.make('Swimmer-v2')
#             self.num_rollouts = 5
#             self.render  = False
#             self.tracker = tracker('Swimmer')
#             self.iteration = 1000

#             
#         
#         def __call__(self, x):
#             self.counter += 1
#             assert len(x) == self.dims
#             assert x.ndim == 1
#             ##assert np.all(x <= self.ub) and np.all(x >= self.lb)
#             M = x.reshape(self.policy.shape)
#             
#             returns = []
#             observations = []
#             actions = []
#             
#             for i in range(self.num_rollouts):
#                 obs    = self.env.reset()
#                 done   = False
#                 totalr = 0.
#                 steps  = 0
#                 while not done:
#                     action = np.dot(M, (obs - self.mean)/self.std)
#                     observations.append(obs)
#                     actions.append(action)
#                     obs, r, done, _ = self.env.step(action)
#                     totalr += r
#                     steps += 1
#                     if self.render:
#                         self.env.render()
#                 returns.append(totalr)
#             final_result = np.mean(returns)*-1
#             self.tracker.track( final_result ) 
#             
#             if self.counter > self.iteration:
#                 os._exit(1)
#             
#             return final_result
#             
#     class HalfCheetah:
#         
#         def __init__(self):
#             lin_policy   = np.load("./MujucoPolicies/HalfCheetah-v1/lin_policy_plus.npz", allow_pickle = True)
#             lin_policy   = lin_policy['arr_0']
#             self.policy  = lin_policy[0]
#             self.mean    = lin_policy[1]
#             self.std     = lin_policy[2]
#             self.dims    = len( self.policy.ravel() )
#             self.lb      = -1 * np.ones(self.dims)
#             self.ub      =  1 * np.ones(self.dims)
#             self.counter = 0
#             self.env          = gym.make('HalfCheetah-v2')
#             self.num_rollouts = 5
#             self.render  = False
#             self.tracker = tracker('HalfCheetah')
#             self.iteration = 5000

#             
#             print("max-min:", np.max(self.policy.ravel() ), np.min(self.policy.ravel() ), len(self.policy.ravel() ) )
#             
#         
#         def __call__(self, x):
#             self.counter += 1
#             assert len(x) == self.dims
#             assert x.ndim == 1
#             ##assert np.all(x <= self.ub) and np.all(x >= self.lb)
#             
#             M = x.reshape(self.policy.shape)
#             
#             returns = []
#             observations = []
#             actions = []
#             
#             for i in range(self.num_rollouts):
#                 obs    = self.env.reset()
#                 done   = False
#                 totalr = 0.
#                 steps  = 0
#                 while not done:
#                     action = np.dot(M, (obs - self.mean)/self.std)
#                     observations.append(obs)
#                     actions.append(action)
#                     obs, r, done, _ = self.env.step(action)
#                     totalr += r
#                     steps += 1
#                     if self.render:
#                         self.env.render()
#                     # print(env.spec)
#                     # if steps % 100 == 0: print("%i/%i"%(steps, env.spec.timestep_limit))
#                     #if steps >= env.spec.timestep_limit:
#                     #    break
#                 returns.append(totalr)
#             final_result = np.mean(returns)*-1
#             self.tracker.track( final_result ) 
#             
#             if self.counter > self.iteration:
#                 os._exit(1)
#             
#             return final_result
#         
#     class Hopper:
#         
#         def __init__(self):
#             lin_policy   = np.load("./MujucoPolicies/Hopper-v1/lin_policy_plus.npz", allow_pickle = True)
#             lin_policy   = lin_policy['arr_0']
#             self.policy  = lin_policy[0]
#             self.mean    = lin_policy[1]
#             self.std     = lin_policy[2]
#             self.dims    = len( self.policy.ravel() )
#             self.lb      = -1.4 * np.ones(self.dims)
#             self.ub      =  1.4 * np.ones(self.dims)
#             self.counter = 0
#             self.env          = gym.make('Hopper-v2')
#             self.num_rollouts = 5
#             self.render  = False
#             self.tracker = tracker('Hopper')
#             self.iteration = 4000

#             
#             print("max-min:", np.max(self.policy.ravel() ), np.min(self.policy.ravel() ) )
#         
#         def __call__(self, x):
#             self.counter += 1
#             assert len(x) == self.dims
#             assert x.ndim == 1
#             # #assert np.all(x <= self.ub) and np.all(x >= self.lb)
#             
#             M = x.reshape(self.policy.shape)
#             
#             returns = []
#             observations = []
#             actions = []
#             
#             for i in range(self.num_rollouts):
#                 obs    = self.env.reset()
#                 done   = False
#                 totalr = 0.
#                 steps  = 0
#                 while not done:
#                     # M      = self.policy
#                     action = np.dot(M, (obs - self.mean)/self.std)
#                     observations.append(obs)
#                     actions.append(action)
#                     obs, r, done, _ = self.env.step(action)
#                     totalr += r
#                     steps  += 1
#                     if self.render:
#                         self.env.render()
#                     # print(env.spec)
#                     # if steps % 100 == 0: print("%i/%i"%(steps, env.spec.timestep_limit))
#                     #if steps >= env.spec.timestep_limit:
#                     #    break
#                 returns.append(totalr)
#             final_result = np.mean(returns)*-1
#             self.tracker.track( final_result )
#             
#             if self.counter > self.iteration:
#                 os._exit(1)
#             
#             return final_result
#             
#     class Walker2d:
#         #params-102
#         def __init__(self):
#             lin_policy   = np.load("./MujucoPolicies/Walker2d-v1/gait5_reward_11200.npz", allow_pickle = True)
#             lin_policy   = lin_policy['arr_0']
#             self.policy  = lin_policy[0]
#             self.mean    = lin_policy[1]
#             self.std     = lin_policy[2]
#             self.dims    = len( self.policy.ravel() )
#             self.lb      = -1.8 * np.ones(self.dims)
#             self.ub      =  0.9 * np.ones(self.dims)
#             self.counter = 0
#             self.env          = gym.make('Walker2d-v2')
#             self.num_rollouts = 5
#             self.render  = False
#             self.tracker = tracker('Walker2d')
#             print("max-min:", np.max(self.policy.ravel() ), np.min(self.policy.ravel() ) )
#             print("parms:", self.policy.shape)
#             self.iteration = 40000
#             
#             
#         
#         def __call__(self, x):
#             self.counter += 1
#             assert len(x) == self.dims
#             assert x.ndim == 1
#             ##assert np.all(x <= self.ub) and np.all(x >= self.lb)
#             
#             M = x.reshape(self.policy.shape)
#             
#             returns = []
#             observations = []
#             actions = []
#             
#             for i in range(self.num_rollouts):
#                 obs    = self.env.reset()
#                 done   = False
#                 totalr = 0.
#                 steps  = 0
#                 while not done:
#                     # M      = self.policy
#                     action = np.dot(M, (obs - self.mean)/self.std)
#                     observations.append(obs)
#                     actions.append(action)
#                     obs, r, done, _ = self.env.step(action)
#                     totalr += r
#                     steps  += 1
#                     if self.render:
#                         self.env.render()
#                     # print(env.spec)
#                     # if steps % 100 == 0: print("%i/%i"%(steps, env.spec.timestep_limit))
#                     #if steps >= env.spec.timestep_limit:
#                     #    break
#                 returns.append(totalr)
#             final_result = np.mean(returns)*-1
#             self.tracker.track( final_result )
#             
#             if self.counter > self.iteration:
#                 os._exit(1)
#             
#             return final_result
#             

#     class Humanoid:
#         #params-6392
#         def __init__(self):
#             lin_policy   = np.load("./MujucoPolicies/Humanoid-v1/policy_reward_11600/lin_policy_plus.npz", allow_pickle = True)
#             lin_policy   = lin_policy['arr_0']
#             print("policy shape:", lin_policy.shape)
#             self.policy  = lin_policy[0]
#             self.mean    = lin_policy[1]
#             self.std     = lin_policy[2]
#             self.dims    = len( self.policy.ravel() )
#             self.lb      = -1 * np.ones(self.dims)
#             self.ub      =  1 * np.ones(self.dims)
#             self.counter = 0
#             self.env          = gym.make('Humanoid-v2')
#             self.num_rollouts = 5
#             self.render  = False
#             print("max-min:", np.max(self.policy.ravel() ), np.min(self.policy.ravel() ) )
#             print("parms:", self.policy.shape)
#             self.tracker = tracker('Humanoid')
#             self.iteration = 40000

#             
#             
#         
#         def __call__(self, x):
#             self.counter += 1
#             assert len(x) == self.dims
#             assert x.ndim == 1
#             ##assert np.all(x <= self.ub) and np.all(x >= self.lb)
#             
#             M = x.reshape(self.policy.shape)
#             
#             returns = []
#             observations = []
#             actions = []
#             
#             for i in range(self.num_rollouts):
#                 obs    = self.env.reset()
#                 done   = False
#                 totalr = 0.
#                 steps  = 0
#                 while not done:
#                     # M      = self.policy
#                     action = np.dot(M, (obs - self.mean)/self.std)
#                     observations.append(obs)
#                     actions.append(action)
#                     obs, r, done, _ = self.env.step(action)
#                     totalr += r
#                     steps  += 1
#                     if self.render:
#                         self.env.render()
#                     # print(env.spec)
#                     # if steps % 100 == 0: print("%i/%i"%(steps, env.spec.timestep_limit))
#                     #if steps >= env.spec.timestep_limit:
#                     #    break
#                 returns.append(totalr)
#             final_result = np.mean(returns)*-1
#             self.tracker.track( final_result )
#             
#             if self.counter > self.iteration:
#                 os._exit(1)
#             
#             return final_result
#         
#         
#         
#         
#         
#         
#         
#         
#         
#         
#         
#         
#         
#         
#         
#         
#         
#         
#         
#         
#         
#         
#         
#         
#         
#         
#         
#         
#         
#         
