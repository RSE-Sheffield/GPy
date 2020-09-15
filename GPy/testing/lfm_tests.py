# This will become a regression test for LFMs.

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


    #k = GPy.kern.MultioutputKern([k1, k2])

    #bob = k.K(x,x)

    #print(k)



#Xt,_,_ = GPy.util.multioutput.build_XY([self.X, self.X])
#X2t,_,_ = GPy.util.multioutput.build_XY([self.X2, self.X2])
#self.assertTrue(check_kernel_gradient_functions(k, X=Xt, X2=X2t, verbose=verbose, fixed_X_dims=-1))