# This is *very* work in progress!

import pickle

import numpy as np

import GPy
from GPy import kern

# Load baseline / test data

with open('GPy/testing/baseline/toydata_baseline.pkl', "rb") as f:
    toydata_baseline = pickle.load(f)

x = toydata_baseline[0] # Observed x (training set)
x_test = toydata_baseline[1] # Observed x (test set)
x_pred = toydata_baseline[2] # x values over which to predict
y = toydata_baseline[3] # Observed y (training set)
y_test = toydata_baseline[4] # Observed y (test set)

def test_make_LFMXLFM():
    # Test that we can make an LFMXLFM kernel
    t_lfmxlfm = kern.LFMXLFM(input_dim = 1 , output_dim = 1)

def test_K_LFMXLFM():
    # Test that this kernel can produce a covariance matrix
    x_eg=np.atleast_2d(np.linspace(-1.,1.,9)).transpose()
    t_lfmxlfm = kern.LFMXLFM(input_dim = 1 , output_dim = 1)
    t_lfmxlfm.K(x_eg,x_eg)

def test_multi_LFM():
    k_lfmxlfm = [kern.LFMXLFM(input_dim = 1 , output_dim = 1) for i in range(9)]

    cov_dict = {(0,0): k_lfmxlfm[0],
                (0,1): k_lfmxlfm[1],
                (0,2): k_lfmxlfm[2],
                (1,0): k_lfmxlfm[3],
                (1,1): k_lfmxlfm[4],
                (1,2): k_lfmxlfm[5],
                (2,0): k_lfmxlfm[6],
                (2,1): k_lfmxlfm[7],
                (2,2): k_lfmxlfm[8]}

    k = GPy.kern.MultioutputKern(k_lfmxlfm, cross_covariances=cov_dict)
    #k = GPy.kern.MultioutputKern([k1, k2])

    #bob = k.K(x,x)

    #print(k)



#Xt,_,_ = GPy.util.multioutput.build_XY([self.X, self.X])
#X2t,_,_ = GPy.util.multioutput.build_XY([self.X2, self.X2])
#self.assertTrue(check_kernel_gradient_functions(k, X=Xt, X2=X2t, verbose=verbose, fixed_X_dims=-1))