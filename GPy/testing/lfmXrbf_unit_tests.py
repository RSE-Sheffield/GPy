import GPy
from GPy.kern.src.lfm import *
import numpy as np
import sys
import unittest
import scipy.io as sio

# load values from matlab (gpmat)

baseline = sio.loadmat('GPy/testing/baseline/baseline.mat')

X = baseline.get('X').flatten()

cov_lfmXrbf = baseline.get('cov_lfmXlfm')

def test_covariance():
    k = GPy.kern.LFMXRBF(input_dim = 1)
    result = k.K(np.atleast_2d(X).transpose())
    np.testing.assert_array_almost_equal(result, cov_lfmXrbf)