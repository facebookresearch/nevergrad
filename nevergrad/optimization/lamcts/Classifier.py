# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# verify
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy as cp

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.svm import SVC
from torch.quasirandom import SobolEngine

from .turbo_1.turbo_1 import Turbo1


# the input will be samples!
class Classifier():
    def __init__(self, samples, dims):
        self.training_counter = 0
        assert dims >= 1
        assert type(samples)  ==  type([])
        self.dims    =   dims
        
        #create a gaussian process regressor
        noise        =   0.1
        m52          =   ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.gpr     =   GaussianProcessRegressor(kernel=m52, alpha=noise**2) #default to CPU
        self.kmean   =   KMeans(n_clusters=2, max_iter = 500)
        #learned boundary
        self.svm     =   SVC(kernel = 'linear') # gamma is stable at scale
        
        #data structures to store
        self.samples = []
        self.X       = np.array([])
        self.fX      = np.array([])
        
        #good region is labeled as zero
        #bad  region is labeled as one
        self.good_label_mean  = -1
        self.bad_label_mean   = -1
        
        self.update_samples(samples)
    
    def is_splittable_svm(self):
        plabel = self.learn_clusters()
        self.learn_boundary(plabel)
        svm_label = self.svm.predict( self.X )
        if len( np.unique(svm_label) ) == 1:
            return False
        else:
            return True
        
    def get_max(self):
        return np.max(self.fX)
        
    def plot_samples_and_boundary(self, func, name):
        assert func.dims == 2
        
        plabels   = self.svm.predict( self.X )
        good_counts = len( self.X[np.where( plabels == 0 )] )
        bad_counts  = len( self.X[np.where( plabels == 1 )] )
        good_mean = np.mean( self.fX[ np.where( plabels == 0 ) ] )
        bad_mean  = np.mean( self.fX[ np.where( plabels == 1 ) ] )
        
        if np.isnan(good_mean) == False and np.isnan(bad_mean) == False:
            assert good_mean > bad_mean

        lb = func.lb
        ub = func.ub
        x = np.linspace(lb[0], ub[0], 100)
        y = np.linspace(lb[1], ub[1], 100)
        xv, yv = np.meshgrid(x, y)
        true_y = []
        for row in range(0, xv.shape[0]):
            for col in range(0, xv.shape[1]):
                x = xv[row][col]
                y = yv[row][col]
                true_y.append( func( np.array( [x, y] ) ) )
        true_y = np.array( true_y )
        pred_labels = self.svm.predict( np.c_[xv.ravel(), yv.ravel()] )
        pred_labels = pred_labels.reshape( xv.shape )
        
        fig, ax = plt.subplots()
        ax.contour(xv, yv, true_y.reshape(xv.shape), cmap=cm.coolwarm)
        ax.contourf(xv, yv, pred_labels, alpha=0.4)
        
        ax.scatter(self.X[ np.where(plabels == 0) , 0 ], self.X[ np.where(plabels == 0) , 1 ], marker='x', label="good-"+str(np.round(good_mean, 2))+"-"+str(good_counts) )
        ax.scatter(self.X[ np.where(plabels == 1) , 0 ], self.X[ np.where(plabels == 1) , 1 ], marker='x', label="bad-"+str(np.round(bad_mean, 2))+"-"+str(bad_counts)    )
        ax.legend(loc="best")
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        plt.savefig(name)
        plt.close()
    
    def get_mean(self):
        return np.mean(self.fX)
        
    def update_samples(self, latest_samples):
        assert type(latest_samples) == type([])
        X  = []
        fX  = []
        for sample in latest_samples:
            X.append(  sample[0] )
            fX.append( sample[1] )
        
        self.X          = np.asarray(X, dtype=np.float32).reshape(-1, self.dims)
        self.fX         = np.asarray(fX,  dtype=np.float32).reshape(-1)
        self.samples    = latest_samples       
        
    def train_gpr(self, samples):
        X  = []
        fX  = []
        for sample in samples:
            X.append(  sample[0] )
            fX.append( sample[1] )
        X  = np.asarray(X).reshape(-1, self.dims)
        fX = np.asarray(fX).reshape(-1)
        
        print("training GPR with ", len(X), " data X")        
        self.gpr.fit(X, fX)
    
    ###########################
    # BO sampling with EI
    ###########################
    
        
    def expected_improvement(self, X, xi=0.0001, use_ei = True):
        ''' Computes the EI at points X based on existing samples X_sample and Y_sample using a Gaussian process surrogate model.
        Args: X: Points at which EI shall be computed (m x d). X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1). gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
        Returns: Expected improvements at points X. '''
        X_sample = self.X
        Y_sample = self.fX.reshape((-1, 1))
        
        gpr = self.gpr
        
        mu, sigma = gpr.predict(X, return_std=True)
        
        if not use_ei:
            return mu
        else:
            #calculate EI
            mu_sample = gpr.predict(X_sample)
            sigma = sigma.reshape(-1, 1)
            mu_sample_opt = np.max(mu_sample)
            with np.errstate(divide='warn'):
                imp = mu - mu_sample_opt - xi
                imp = imp.reshape((-1, 1))
                Z = imp / sigma
                ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma == 0.0] = 0.0
            return ei
            
    def plot_boundary(self, X):
        if X.shape[1] > 2:
            return
        fig, ax = plt.subplots()
        ax.scatter( X[ :, 0 ], X[ :, 1 ] , marker='.')
        ax.scatter(self.X[ : , 0 ], self.X[ : , 1 ], marker='x')
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        plt.savefig("boundary.pdf")
        plt.close()
    
    def get_sample_ratio_in_region( self, cands, path ):
        total = len(cands)
        for node in path:
            boundary = node[0].classifier.svm
            if len(cands) == 0:
                return 0, np.array([])
            assert len(cands) > 0
            cands = cands[ boundary.predict( cands ) == node[1] ] 
            # node[1] store the direction to go
        ratio = len(cands) / total
        assert len(cands) <= total
        return ratio, cands

    def propose_rand_samples_probe(self, nums_samples, path, lb, ub):

        seed   = np.random.randint(int(1e6))
        sobol  = SobolEngine(dimension = self.dims, scramble=True, seed=seed)

        center = np.mean(self.X, axis = 0)
        #check if the center located in the region
        ratio, tmp = self.get_sample_ratio_in_region( np.reshape(center, (1, len(center) ) ), path )
        if ratio == 0:
            print("==>center not in the region, using random samples")
            return self.propose_rand_samples(nums_samples, lb, ub)
        # it is possible that the selected region has no points,
        # so we need check here

        axes    = len( center )
        
        final_L = []
        for axis in range(0, axes):
            L       = np.zeros( center.shape )
            L[axis] = 0.01
            ratio   = 1
            
            while ratio >= 0.9:
                L[axis] = L[axis]*2
                if L[axis] >= (ub[axis] - lb[axis]):
                    break
                lb_     = np.clip( center - L/2, lb, ub )
                ub_     = np.clip( center + L/2, lb, ub )
                cands_  = sobol.draw(10000).to(dtype=torch.float64).cpu().detach().numpy()
                cands_  = (ub_ - lb_)*cands_ + lb_
                ratio, tmp = self.get_sample_ratio_in_region(cands_, path )
                # print("ratio:", ratio, L[axis])
            final_L.append( L[axis] )
            #print("axis:", axis," L:", L," ratio:", ratio)
            #print("lb:", lb_)
            #print("ub:", ub_)
        final_L   = np.array( final_L )
        lb_       = np.clip( center - final_L/2, lb, ub )
        ub_       = np.clip( center + final_L/2, lb, ub )
        print("center:", center)
        print("final lb:", lb_)
        print("final ub:", ub_)
    
        count         = 0
        cands         = np.array([])
        while len(cands) < 10000:
            count    += 10000
            cands     = sobol.draw(count).to(dtype=torch.float64).cpu().detach().numpy()
        
            cands     = (ub_ - lb_)*cands + lb_
            ratio, cands = self.get_sample_ratio_in_region(cands, path)
            samples_count = len( cands )
        
        #extract candidates 
        
        return cands
            
    def propose_rand_samples_sobol(self, nums_samples, path, lb, ub):
        
        #rejected sampling
        selected_cands = np.zeros((1, self.dims))
        seed   = np.random.randint(int(1e6))
        sobol  = SobolEngine(dimension = self.dims, scramble=True, seed=seed)
        
        # scale the samples to the entire search space
        # ----------------------------------- #
        # while len(selected_cands) <= nums_samples:
        #     cands  = sobol.draw(100000).to(dtype=torch.float64).cpu().detach().numpy()
        #     cands  = (ub - lb)*cands + lb
        #     for node in path:
        #         boundary = node[0].classifier.svm
        #         if len(cands) == 0:
        #             return []
        #         cands = cands[ boundary.predict(cands) == node[1] ] # node[1] store the direction to go
        #     selected_cands = np.append( selected_cands, cands, axis= 0)
        #     print("total sampled:", len(selected_cands) )
        # return cands
        # ----------------------------------- #
        #shrink the cands region
        
        ratio_check, centers = self.get_sample_ratio_in_region(self.X, path)
        # no current samples located in the region
        # should not happen
        print("ratio check:", ratio_check, len(self.X) )
        # assert ratio_check > 0
        if ratio_check == 0 or len(centers) == 0:
            return self.propose_rand_samples( nums_samples, lb, ub )
        
        lb_    = None
        ub_    = None
        
        final_cands = []
        print("###total centers:", len(centers) )
        for center in centers:
            center = self.X[ np.random.randint( len(self.X) ) ]
            cands  = sobol.draw(2000).to(dtype=torch.float64).cpu().detach().numpy()
            ratio  = 1
            L      = 0.0001
            Blimit = np.max(ub - lb)
            
            #print("center:", center, len( center ), L, ratio )
            while ratio == 1 and L < Blimit:                    
                lb_    = np.clip( center - L/2, lb, ub )
                ub_    = np.clip( center + L/2, lb, ub )
                # print(ratio)
                cands_ = cp.deepcopy( cands )
                cands_ = (ub_ - lb_)*cands_ + lb_
                ratio, cands_ = self.get_sample_ratio_in_region(cands_, path)
                if ratio < 1:
                    #print("final ratio:", ratio)
                    #print("lb_:", np.sum( lb_ )/len( lb_ ) )
                    #print("ub_:", np.sum( ub_ )/len( ub_ ) )
                    final_cands.extend( cands_.tolist() )
                L = L*2
        print("=====>final cands:", len(final_cands) )
        final_cands      = np.array( final_cands )
        if len(final_cands) > nums_samples:
            final_cands_idx  = np.random.choice( len(final_cands), nums_samples )
            return final_cands[final_cands_idx]
        else:
            if len(final_cands) == 0:
                return self.propose_rand_samples( nums_samples, lb, ub )
            else:
                return final_cands
        
    def propose_samples_bo( self, nums_samples = 10, path = None, lb = None, ub = None, samples = None):
        ''' Proposes the next sampling point by optimizing the acquisition function. 
        Args: acquisition: Acquisition function. X_sample: Sample locations (n x d). 
        Y_sample: Sample values (n x 1). gpr: A GaussianProcessRegressor fitted to samples. 
        Returns: Location of the acquisition function maximum. '''
        assert path is not None and len(path) >= 0
        assert lb is not None and ub is not None
        assert samples is not None and len(samples) > 0
        
        self.train_gpr( samples ) # learn in unit cube
        
        dim  = self.dims
        nums_rand_samples = 10000
        if len(path) == 0:
            return self.propose_rand_samples(nums_samples, lb, ub)
        
        X    = self.propose_rand_samples_sobol(nums_rand_samples, path, lb, ub)
        print("samples in the region:", len(X) )
        # self.plot_boundary(X)
        if len(X) == 0:
            return self.propose_rand_samples(nums_samples, lb, ub)
        
        X_ei = self.expected_improvement(X, xi=0.001, use_ei = True)
        row, col = X.shape
    
        X_ei = X_ei.reshape(len(X))
        n = nums_samples
        if X_ei.shape[0] < n:
            n = X_ei.shape[0]
        indices = np.argsort(X_ei)[-n:]
        proposed_X = X[indices]
        return proposed_X
        
    ###########################
    # sampling with turbo
    ###########################
    # version 1: select a partition, perform one-time turbo search
        
    def propose_samples_turbo(self, num_samples, path, func, lb, ub, device):
        #throw a uniform sampling in the selected partition
        X_init = self.propose_rand_samples_sobol(30, path, lb, ub)
        #get samples around the selected partition
        print("sampled ", len(X_init), " for the initialization")
        turbo1 = Turbo1(
            f  = func,              # Handle to objective function
            lb = lb,           # Numpy array specifying lower bounds
            ub = ub,           # Numpy array specifying upper bounds
            n_init = 30,            # Number of initial bounds from an Latin hypercube design
            max_evals  = num_samples, # Maximum number of evaluations
            batch_size = 1,         # How large batch size TuRBO uses
            verbose=True,           # Print information from each batch
            use_ard=True,           # Set to true if you want to use ARD for the GP kernel
            max_cholesky_size=2000, # When we switch from Cholesky to Lanczos
            n_training_steps=50,    # Number of steps of ADAM to learn the hypers
            min_cuda= 40,          #  Run on the CPU for small datasets
            device=device,           # "cpu" or "cuda"
            dtype="float64",        # float64 or float32
            boundary=path,
            X_init = X_init
        )

        proposed_X, fX = turbo1.optimize( )
        fX = fX*-1

        return proposed_X, fX
        
    
            
    ###########################
    # random sampling
    ###########################
    
    def propose_rand_samples(self, nums_samples, lb, ub):
        x = np.random.uniform(lb, ub, size = (nums_samples, self.dims) )
        return x
        
        
    def propose_samples_rand( self, nums_samples = 10):
        return self.propose_rand_samples(nums_samples, self.lb, self.ub)
                
    ###########################
    # learning boundary
    ###########################
    
        
    def get_cluster_mean(self, plabel):
        assert plabel.shape[0] == self.fX.shape[0] 
        
        zero_label_fX = []
        one_label_fX  = []
        
        for idx in range(0, len(plabel)):
            if plabel[idx] == 0:
                zero_label_fX.append( self.fX[idx]  )
            elif plabel[idx] == 1:
                one_label_fX.append( self.fX[idx] )
            else:
                print("kmean should only predict two clusters, Classifiers.py:line73")
                os._exit(1)
                
        good_label_mean = np.mean( np.array(zero_label_fX) )
        bad_label_mean  = np.mean( np.array(one_label_fX) )
        return good_label_mean, bad_label_mean
        
    def learn_boundary(self, plabel):
        assert len(plabel) == len(self.X)
        self.svm.fit(self.X, plabel)
        
    def learn_clusters(self):
        assert len(self.samples) >= 2, "samples must > 0"
        assert self.X.shape[0], "points must > 0"
        assert self.fX.shape[0], "fX must > 0"
        assert self.X.shape[0] == self.fX.shape[0]
        
        tmp = np.concatenate( (self.X, self.fX.reshape([-1, 1]) ), axis = 1 )
        assert tmp.shape[0] == self.fX.shape[0]
        
        self.kmean  = self.kmean.fit(tmp)
        plabel      = self.kmean.predict( tmp )
        
        # the 0-1 labels in kmean can be different from the actual
        # flip the label is not consistent
        # 0: good cluster, 1: bad cluster
        
        self.good_label_mean , self.bad_label_mean = self.get_cluster_mean(plabel)
        
        if self.bad_label_mean > self.good_label_mean:
            for idx in range(0, len(plabel)):
                if plabel[idx] == 0:
                    plabel[idx] = 1
                else:
                    plabel[idx] = 0
                    
        self.good_label_mean , self.bad_label_mean = self.get_cluster_mean(plabel)
        
        return plabel
        
    def split_data(self):
        good_samples = []
        bad_samples  = []
        train_good_samples = []
        train_bad_samples  = []
        if len( self.samples ) == 0:
            return good_samples, bad_samples
        
        plabel = self.learn_clusters( )
        self.learn_boundary( plabel )
        
        for idx in range(0, len(plabel)):
            if plabel[idx] == 0:
                #ensure the consistency
                assert self.samples[idx][-1] - self.fX[idx] <= 1
                good_samples.append( self.samples[idx] )
                train_good_samples.append( self.X[idx] )
            else:
                bad_samples.append( self.samples[idx] )
                train_bad_samples.append( self.X[idx] )
        
        train_good_samples = np.array(train_good_samples)
        train_bad_samples  = np.array(train_bad_samples)
                        
        assert len(good_samples) + len(bad_samples) == len(self.samples)
        
        #TODO: make sure the whole search space will be covered
        
        return  good_samples, bad_samples



    
    
    

