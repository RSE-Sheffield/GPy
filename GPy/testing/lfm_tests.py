# This will become a regression test for LFMs.

import os
import pickle

import matplotlib.pyplot as plt

import numpy as np

import GPy
from GPy import kern

basedir = os.path.dirname(os.path.relpath(os.path.abspath(__file__)))
result_dir = os.path.join(basedir, 'testresult','.')
baseline_dir = os.path.join(basedir, 'baseline','.')

# Load baseline / test data

with open(os.path.join(baseline_dir, "toydata_baseline.pkl"), "rb") as f:
    toydata_baseline = pickle.load(f)

x = toydata_baseline[0] # Observed x (training set)
x_test = toydata_baseline[1] # Observed x (test set)
x_pred = toydata_baseline[2] # x values over which to predict
y = toydata_baseline[3] # Observed y (training set)
y_test = toydata_baseline[4] # Observed y (test set)

def test_lfmxlfm_update_gradients_full():
    #this test duplicates part of `check_kernel_gradient_functions()` and should be removed

    k = GPy.kern.LFMXLFM(input_dim = 1 , output_dim = 1)
    
    #k = GPy.kern.RBF(input_dim = 1) #this works

    X = np.random.randn(10, k.input_dim)

    dL_dK = np.random.rand(X.shape[0], X.shape[0])
    
    k.update_gradients_full(dL_dK, X)

def test_multioutput_optimisation():

    # build kernel

    k_rbf = [GPy.kern.RBF(input_dim = 1) for i in range(9)]
    
    cov_dict = {(0,0): k_rbf[0],
                    (0,1): k_rbf[1],
                    (0,2): k_rbf[2],
                    (1,0): k_rbf[3],
                    (1,1): k_rbf[4],
                    (1,2): k_rbf[5],
                    (2,0): k_rbf[6],
                    (2,1): k_rbf[7],
                    (2,2): k_rbf[8]}
    
    Xm, Ym, Im = GPy.util.multioutput.build_XY(x, y)
   
    # build model
   
    likelihoods = [GPy.likelihoods.Gaussian(variance=0.1) for i in range(9)]

    m = GPy.models.MultioutputGP(X_list = x, Y_list = y, kernel_list = k_rbf, likelihood_list = likelihoods, kernel_cross_covariances = cov_dict)

    # plot initial covariance matrix

    cov = m.kern.K(Xm, Ym)
    plt.imshow(cov)

    # save image for manual check      
    
    plt.savefig(os.path.join(result_dir, "covplot_pre.png"))

    # randomise model

    # m.randomize() # this causes the test to error

    # optimise model

    m.optimize()

    # plot post-optimisation covariance matrix

    cov = m.kern.K(Xm, Ym)
    plt.imshow(cov)

    # save image for manual check      
    
    plt.savefig(os.path.join(result_dir, "covplot_post.png"))