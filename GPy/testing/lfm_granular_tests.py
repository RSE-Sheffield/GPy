import GPy
from GPy.kern.src.lfm import *
import numpy as np
import sys
import unittest
import scipy.io as sio

# load values from matlab (gpmat)

baseline = sio.loadmat('GPy/testing/baseline/baseline.mat')

X = baseline.get('X').flatten()
gamma = baseline.get('gamma')
sigma2 = baseline.get('sigma2')
preconst = baseline.get('preconst').flatten()
preconst2 = baseline.get('preconst2').flatten()
pregamma = baseline.get('pregamma').flatten()
pregamma2 = baseline.get('pregamma2').flatten()
preexp = baseline.get('preexp')
cov = baseline.get('cov')
covGrad = baseline.get('covGrad')

gradthetagamma1 = baseline.get('gradthetagamma1').flatten()

# Values copied from matlab defaults

inversewidth = 1
mass = 1
spring = 1
damper = 1
sensitivity = 1
alpha = 0.5
omega = 0.866025403784439
gamma = 0.5 + 0.866025403784439j
variance = 1
inversewidth = 1
sigma2 = 2 / inversewidth

# Make a kernel

k = GPy.kern.LFMXLFM(input_dim = 1)

# Check parameters are the same as matlab

def test_parameters():
    
    #assert(inversewidth == k.scale[0])
    np.testing.assert_array_almost_equal(np.array([mass,mass]), k.mass)
    np.testing.assert_array_almost_equal(np.array([spring,spring]), k.spring)
    np.testing.assert_array_almost_equal(np.array([damper,damper]), k.damper)
    np.testing.assert_array_almost_equal(np.array([sensitivity,sensitivity]), k.sensitivity)

    np.testing.assert_array_almost_equal(np.array([alpha,alpha]), k.alpha)
    np.testing.assert_array_almost_equal(np.array([omega,omega]), k.omega)
    np.testing.assert_array_almost_equal(np.array([gamma,gamma]), k.gamma)
    np.testing.assert_array_almost_equal(np.array([sigma2,sigma2]), k.sigma2)

# Test helper functions

def test_lfmUpsilonMatrix():
    result = lfmUpsilonMatrix(gamma, sigma2, X, X)
    np.testing.assert_array_almost_equal(result, baseline.get('baseline_upsilonmatrix'))

def test_lfmUpsilonVector():
    result = lfmUpsilonVector(gamma, sigma2, X)
    np.testing.assert_array_almost_equal(result, baseline.get('baseline_upsilonvector').flatten())

def test_lfmGradientUpsilonMatrix():
    result = lfmGradientUpsilonMatrix(gamma, sigma2, X, X)
    np.testing.assert_array_almost_equal(result, baseline.get('baseline_gradientupsilonmatrix'))

def test_lfmGradientUpsilonVector():
    result = lfmGradientUpsilonVector(gamma, sigma2, X)
    np.testing.assert_array_almost_equal(result, baseline.get('baseline_gradientupsilonvector').flatten())

def test_lfmGradientSigmaUpsilonMatrix():
    result = lfmGradientSigmaUpsilonMatrix(gamma, sigma2, X, X)
    np.testing.assert_array_almost_equal(result, baseline.get('baseline_gradientsigmaupsilonmatrix'))

def test_lfmGradientSigmaUpsilonVector():
    result = lfmGradientSigmaUpsilonVector(gamma, sigma2, X)
    np.testing.assert_array_almost_equal(result, baseline.get('baseline_gradientsigmaupsilonvector').flatten())

def test_lfmComputeH3():
    result = lfmComputeH3(gamma, gamma, sigma2, X, X, preconst, mode = False, term = True)[0]
    np.testing.assert_array_almost_equal(result, baseline.get('baseline_computeH3'))

def test_lfmComputeH4():
    result = lfmComputeH4(gamma, gamma, sigma2, X, pregamma, preexp, mode = False, term = True)[0]
    np.testing.assert_array_almost_equal(result, baseline.get('baseline_computeH4'))

def test_lfmGradientH31():
    result = lfmGradientH31(preconst,
                            preconst2,
                            gradthetagamma1,
                            baseline.get('baseline_gradientupsilonmatrix'),
                            1,
                            baseline.get('baseline_upsilonmatrix'),
                            1,
                            mode = False,
                            term = True)
    np.testing.assert_array_almost_equal(result, baseline.get('baseline_gradientH31'))                  

def test_lfmGradientH32():
    result = lfmGradientH32(pregamma2,
                            gradthetagamma1,
                            baseline.get('baseline_upsilonmatrix'),
                            1,
                            mode = False,
                            term = False)
    np.testing.assert_array_almost_equal(result, baseline.get('baseline_gradientH32'))  

# ToDo:

# lfmGradientH41
# lfmGradientH42
# lfmGradientSigmaH3
# lfmGradientSigmaH4

# Check matlab and python produce the same covariance matrix

def test_covariance():
    result = k.K(np.atleast_2d(X).transpose())
    np.testing.assert_array_almost_equal(result, cov)

# Check matlab and python produce the same gradients

def test_gradient():
    grad1 = baseline.get('grad1')
    grad2 = baseline.get('grad2')
    k.update_gradients_full(covGrad, np.atleast_2d(X).transpose(), np.atleast_2d(X).transpose())
    # ToDo figure out equivalence (below is a placeholder)
    result=[k.mass.gradient]    
    np.testing.assert_array_almost_equal(result,covGrad)
