# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# verify
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
import collections
import copy as cp
import math
from collections import OrderedDict
import os.path
import numpy as np
import time
import operator
import sys
import pickle
import os
import random
from datetime import datetime
from .Node import Node
from .Net_Trainer import Net_Trainer
from .utils import latin_hypercube, from_unit_cube
from .functions import *
from torch.quasirandom import SobolEngine
import torch
#from functions.rover_function import rover_nag
#from functions.mujoco_functions import *
# from push_function import PushReward

class MCTS:
    #############################################

    def __init__(self, lb, ub, dims, ninits, func, device):
        self.dims                    =  dims
        self.samples                 =  []
        self.nodes                   =  []
        self.Cp                      =  5
        assert (lb is None) == (ub is None), "One-sided bounds are not yet supported."
        self.lb                      =  lb if lb is not None else -(np.pi/2)*np.ones(dims)
        self.ub                      =  ub if ub is not None else (np.pi/2)*np.ones(dims)
        self.ninits                  =  ninits
        self.func                    =  func if ub is not None and lb is not None else lambda x: func(np.tanh(x))
        self.device = device
        self.curt_best_value         =  float("-inf")
        self.curt_best_sample        =  None
        self.best_value_trace        =  []
        self.sample_counter          =  0
        self.visualization           =  False
        
        self.LEAF_SAMPLE_SIZE        =  100
        
        #we start the most basic form of the tree, 3 nodes and height = 1
        root = Node( parent = None, dims = self.dims, reset_id = True )
        self.nodes.append( root )
        
        self.ROOT = root
        self.CURT = self.ROOT
        self.init_train()
        
    def populate_training_data(self):
        #only keep root
        self.ROOT.obj_counter = 0
        for node in self.nodes:
            node.clear_data()
        self.nodes.clear()
        new_root  = Node(parent = None,   dims = self.dims, reset_id = True )
        self.nodes.append( new_root )
        
        self.ROOT = new_root
        self.CURT = self.ROOT
        self.ROOT.update_bag( self.samples )
    
    def get_leaf_status(self):
        status = []
        for node in self.nodes:
            if node.is_leaf() == True and len(node.bag) > self.LEAF_SAMPLE_SIZE and node.is_svm_splittable == True:
                status.append( True  )
            else:
                status.append( False )
        return np.array( status )
        
    def get_split_idx(self):
        split_by_samples = np.argwhere( self.get_leaf_status() == True ).reshape(-1)
        return split_by_samples
    
    def is_splitable(self):
        status = self.get_leaf_status()
        if True in status:
            return True
        else:
            return False
        
    def dynamic_treeify(self):
        # we bifurcate a node once it contains over 20 samples
        # the node will bifurcate into a good and a bad kid
        self.populate_training_data()
        print("total nodes:", len(self.nodes) )
        assert len(self.ROOT.bag) == len(self.samples)
        assert len(self.nodes)    == 1
        
        print("keep splitting:", self.is_splitable(), self.get_split_idx() )
        
        while self.is_splitable():
            to_split = self.get_split_idx()
            print("==>to split:", to_split, " total:", len(self.nodes) )
            for nidx in to_split:
                parent = self.nodes[nidx] # parent check if the boundary is splittable by svm
                assert len(parent.bag) >= self.LEAF_SAMPLE_SIZE
                assert parent.is_svm_splittable == True
                # print("spliting node:", parent.get_name(), len(parent.bag))
                good_kid_data, bad_kid_data = parent.train_and_split()
                #creat two kids, assign the data, and push into lists
                # children's lb and ub will be decided by its parent
                assert len(good_kid_data) + len(bad_kid_data) == len(parent.bag)
                assert len(good_kid_data) > 0
                assert len(bad_kid_data)  > 0
                good_kid = Node(parent = parent, dims = self.dims, reset_id = False )
                bad_kid  = Node(parent = parent, dims = self.dims, reset_id = False  )
                good_kid.update_bag( good_kid_data )
                bad_kid.update_bag(  bad_kid_data  )
            
                parent.update_kids( good_kid = good_kid, bad_kid = bad_kid )
            
                self.nodes.append(good_kid)
                self.nodes.append(bad_kid)
                
            print("continue split:", self.is_splitable())

            #CAUTION: make sure the good kid in into list first
        
        self.print_tree()
        
    def collect_samples(self, sample, value = None):
        #TODO: to perform some checks here
        if value == None:
            value = self.func(sample)*-1
            
        if value > self.curt_best_value:
            self.curt_best_value  = value
            self.curt_best_sample = sample 
            self.best_value_trace.append( (value, self.sample_counter) )
        self.sample_counter += 1
        if self.sample_counter % 500 == 0:
            self.dump_trace()
        self.samples.append( (sample, value) )
        return value
        
    def init_train(self):
        
        # here we use latin hyper space to generate init samples in the search space
        init_points = latin_hypercube(self.ninits, self.dims)
        init_points = from_unit_cube(init_points, self.lb, self.ub)
        
        for point in init_points:
            self.collect_samples(point)
        
        print("="*10 + 'collect '+ str(len(self.samples) ) +' points for initializing MCTS'+"="*10)
        print("lb:", self.lb)
        print("ub:", self.ub)
        print("Cp:", self.Cp)
        print("inits:", self.ninits)
        print("dims:", self.dims)
        print("="*58)
        
    def print_tree(self):
        print('-'*100)
        for node in self.nodes:
            print(node)
        print('-'*100)

    def reset_to_root(self):
        self.CURT = self.ROOT
    
    def load_agent(self):
        node_path = 'mcts_agent'
        if os.path.isfile(node_path) == True:
            with open(node_path, 'rb') as json_data:
                self = pickle.load(json_data)
                print("=====>loads:", len(self.samples)," samples" )

    def dump_agent(self):
        node_path = 'mcts_agent'
        print("dumping the agent.....")
        with open(node_path,"wb") as outfile:
            pickle.dump(self, outfile)
            
    def dump_samples(self):
        sample_path = 'samples_'+str(self.sample_counter)
        with open(sample_path, "wb") as outfile:
            pickle.dump(self.samples, outfile)
    
    def dump_trace(self):
        trace_path = 'results/result'+str(self.sample_counter)
        final_results_str = json.dumps(self.best_value_trace)
        with open(trace_path, "a") as f:
            f.write(final_results_str + '\n')

    def greedy_select(self):
        self.reset_to_root()
        curt_node = self.ROOT
        path      = [ ]
        if self.visualization == True:
            curt_node.plot_samples_and_boundary(self.func)
        while curt_node.is_leaf() == False:
            UCT = []
            for i in curt_node.kids:
                UCT.append( i.get_xbar() )
            choice = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]
            path.append( (curt_node, choice) )
            curt_node = curt_node.kids[choice]
            if curt_node.is_leaf() == False and self.visualization == True:
                curt_node.plot_samples_and_boundary(self.func)
            print("=>", curt_node.get_name(), end=' ' )
        print("")
        return curt_node, path
        

    def select(self):
        self.reset_to_root()
        curt_node = self.ROOT
        path      = [ ]
        
        while curt_node.is_leaf() == False:
            UCT = []
            for i in curt_node.kids:
                UCT.append( i.get_uct(self.Cp) )
            choice = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]
            path.append( (curt_node, choice) )
            curt_node = curt_node.kids[choice]
            print("=>", curt_node.get_name(), end=' ' )
        return curt_node, path
    
    def backpropogate(self, leaf, acc):
        curt_node = leaf
        while curt_node is not None:
            assert curt_node.n > 0
            curt_node.x_bar = (curt_node.x_bar*curt_node.n + acc) / (curt_node.n + 1)
            curt_node.n    += 1
            curt_node       = curt_node.parent

    def search(self):
        for _ in range(0, 1000):
            self.dynamic_treeify()
            leaf, path = self.select()
            print("selected leaf:", leaf.get_name() )
            print(path)
            for _ in range(0, 1):
                # samples = leaf.propose_samples_bo( 1, path, self.lb, self.ub, self.samples )
                samples, values = leaf.propose_samples_turbo(10000, path, self.func, self.lb, self.ub, device=self.device)
                for idx in range(0, len(samples)):
                    # value = self.collect_samples( samples[idx])
                    value = self.collect_samples( samples[idx], values[idx] )
                    self.backpropogate( leaf, value )
            if len(self.samples) > 100000:
                break
            print("======>curt_best:", self.curt_best_value, self.curt_best_sample )



class TargetFunction:
    def __init__(self, dims, func, lb, ub, budget):
        self.dims    = dims
        self.lb      = lb  #-5.12 * np.ones(dims)
        self.ub      = ub  # 5.12 * np.ones(dims)
        self.counter = 0
        self.iteration = budget
#        self.tracker = ng_tracker('target')
        self.func = func

    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        # #assert np.all(x <= self.ub) and np.all(x >= self.lb)

        tmp = 0;
        result = self.func(x)  #10 * np.sum(x * x)
        #self.tracker.track( result )
        if self.counter > self.iteration:
            raise ValueError("Too many calls to the objective function!")
        return result

def lamcts_minimize(func, dims, budget, lb=None, ub=None, device='cuda'):
    # Here func takes a ndarray in R^dims and outputs a float.
    f = TargetFunction(dims = dims, func=func, lb=lb, ub=ub, budget=budget)
    # f = Ackley(dims = 20)
    # f = Rastrigin(dims = 10)
    # f = Rosenrock(dims = 20)
    # f = Schwefel(dims = 7)
    # f = Zakharov()
    # f = PushReward()
    # f = Lunarlanding()
    # f = Ant()
    # f = HalfCheetah()
    # f = Walker2d()
    # f = Swimmer()
    # f = Hopper()
    agent = MCTS(lb = f.lb, ub = f.ub, dims = f.dims, ninits = 40, func = f, device=device)
    agent.search()
    
