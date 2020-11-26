import GPy
from GPy.kern.src.lfm import *
import numpy as np
import sys
import unittest
import scipy.io as sio

# load values from matlab (gpmat)

baseline = sio.loadmat('GPy/testing/baseline/baseline.mat')

X = baseline.get('X').flatten()

cov_lfmXrbf = baseline.get('cov_lfmXrbf')

covGrad = baseline.get('covGrad')

grad1_lfmXrbf = baseline.get('grad1_lfmXrbf').flatten()
grad2_lfmXrbf = baseline.get('grad2_lfmXrbf').flatten()

k = GPy.kern.LFMXRBF(input_dim = 1)

def test_covariance():    
    result = k.K(np.atleast_2d(X).transpose())
    np.testing.assert_array_almost_equal(result, cov_lfmXrbf)

def test_gradient_mass():  
    k.update_gradients_full(covGrad, np.atleast_2d(X).transpose(), np.atleast_2d(X).transpose())
    np.testing.assert_array_almost_equal(k.mass.gradient, grad1_lfmXrbf[0])

def test_gradient_spring():  
    k.update_gradients_full(covGrad, np.atleast_2d(X).transpose(), np.atleast_2d(X).transpose())
    np.testing.assert_array_almost_equal(k.spring.gradient, grad1_lfmXrbf[1])

def test_gradient_damper():  
    k.update_gradients_full(covGrad, np.atleast_2d(X).transpose(), np.atleast_2d(X).transpose())
    np.testing.assert_array_almost_equal(k.damper.gradient, grad1_lfmXrbf[2])

def test_gradient_scale():  
    k.update_gradients_full(covGrad, np.atleast_2d(X).transpose(), np.atleast_2d(X).transpose())
    np.testing.assert_array_almost_equal(k.scale.gradient, grad1_lfmXrbf[3])

def test_gradient_sensitivity():  
    k.update_gradients_full(covGrad, np.atleast_2d(X).transpose(), np.atleast_2d(X).transpose())
    np.testing.assert_array_almost_equal(k.sensitivity.gradient, grad1_lfmXrbf[4])