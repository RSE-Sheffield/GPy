# This will become a regression test for LFMs.

import os
import pickle

import matplotlib.pyplot as plt

import numpy as np

import scipy.io as sio

import GPy
from GPy import kern

basedir = os.path.dirname(os.path.relpath(os.path.abspath(__file__)))
result_dir = os.path.join(basedir, 'testresult','.')
baseline_dir = os.path.join(basedir, 'baseline','.')

baseline_file = os.path.join(baseline_dir, 'toyDataBatchNLFM1.mat')

# Load baseline / test data
toydata_baseline = sio.loadmat(baseline_file)

# Extract data from `.m` file
x_temp = toydata_baseline['XTemp']
x_test_temp = toydata_baseline['XTestTemp']
y_temp = toydata_baseline['yTemp']
y_test_temp = toydata_baseline['yTestTemp']

# Re-arrange data
x = x_temp[0, 0:3].tolist() # Observed x (training set)
y = y_temp[0, 0:3].tolist() # Observed y (training set)
x_test = x_test_temp[0, 0:3].tolist() # Observed x (test set)
y_test = y_test_temp[0, 0:3].tolist() # Observed y (test set)

def test_lfmxlfm_update_gradients_full():
    #this test duplicates part of `check_kernel_gradient_functions()` and should be removed

    k = GPy.kern.LFMXLFM(input_dim = 1)
    
    #k = GPy.kern.RBF(input_dim = 1) #this works

    X = np.random.randn(10, k.input_dim)

    X2 = np.random.randn(10, k.input_dim)

    dL_dK = np.random.rand(X.shape[0], X.shape[0])
    
    k.update_gradients_full(dL_dK, X, X2)

def test_multioutput_optimisation():

    # build kernel

    k_lfmxlfm = [GPy.kern.LFMXLFM(input_dim = 1) for i in range(9)]
    
    cov_dict = {(0,0): k_lfmxlfm[0],
                    (0,1): k_lfmxlfm[1],
                    (0,2): k_lfmxlfm[2],
                    (1,0): k_lfmxlfm[3],
                    (1,1): k_lfmxlfm[4],
                    (1,2): k_lfmxlfm[5],
                    (2,0): k_lfmxlfm[6],
                    (2,1): k_lfmxlfm[7],
                    (2,2): k_lfmxlfm[8]}
    
    Xm, Ym, Im = GPy.util.multioutput.build_XY(x, y)
   
    # build model
   
    likelihoods = [GPy.likelihoods.Gaussian(variance=0.1) for i in range(9)]

    m = GPy.models.MultioutputGP(X_list = x, Y_list = y, kernel_list = k_lfmxlfm, likelihood_list = likelihoods, kernel_cross_covariances = cov_dict)

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