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
preexpt = baseline.get('preexpt')
cov = baseline.get('cov')
covGrad = baseline.get('covGrad')

gradthetagamma1 = baseline.get('gradthetagamma1').flatten()
gradthetagamma2 = baseline.get('gradthetagamma2')

grad1 = baseline.get('grad1').flatten()
grad2 = baseline.get('grad2').flatten()

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

# Calculate gradients

k.update_gradients_full(covGrad, np.atleast_2d(X).transpose(), np.atleast_2d(X).transpose())

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
                            gradthetagamma2,
                            baseline.get('baseline_gradientupsilonmatrix'),
                            1,
                            baseline.get('baseline_upsilonmatrix'),
                            1,
                            mode = False,
                            term = False)
    np.testing.assert_array_almost_equal(result, baseline.get('baseline_gradientH31'))                  

def test_lfmGradientH32():
    result = lfmGradientH32(pregamma2,
                            gradthetagamma2,
                            baseline.get('baseline_upsilonmatrix'),
                            1,
                            mode = False,
                            term = True)
    np.testing.assert_array_almost_equal(result, baseline.get('baseline_gradientH32'))  

def test_lfmGradientH41():
    result  = lfmGradientH41(pregamma,
                            pregamma2,
                            gradthetagamma2,
                            preexp,
                            baseline.get('baseline_gradientupsilonvector').flatten(),
                            1,
                            baseline.get('baseline_upsilonvector').flatten(),
                            1,
                            mode = False,
                            term = False)
    np.testing.assert_array_almost_equal(result, baseline.get('baseline_gradientH41')) 

def test_lfmGradientH42():
    result = lfmGradientH42(pregamma,
                            pregamma2,
                            gradthetagamma2,
                            preexp,
                            preexpt,
                            baseline.get('baseline_upsilonvector').flatten(),
                            1,
                            mode = False,
                            term = True)
    np.testing.assert_array_almost_equal(result, baseline.get('baseline_gradientH42')) 

def test_lfmGradientSigmaH3():
    result = lfmGradientSigmaH3(gamma,
                                gamma,
                                sigma2,
                                X,
                                X,
                                preconst,
                                mode = False,
                                term = True)
    np.testing.assert_array_almost_equal(result, baseline.get('baseline_gradientSigmaH3')) 

def test_lfmGradientSigmaH4():
    result = lfmGradientSigmaH4(gamma,
                                gamma,
                                sigma2,
                                X,
                                pregamma,
                                preexp,
                                mode = False,
                                term = True)
    np.testing.assert_array_almost_equal(result, baseline.get('baseline_gradientSigmaH4')) 

# ToDo:

# Digonal helpers, cover more of branching logic

# Check matlab and python produce the same covariance matrix

def test_covariance():
    result = k.K(np.atleast_2d(X).transpose())
    np.testing.assert_array_almost_equal(result, cov)

# Check matlab and python produce the same gradients

def test_gradient_mass():  
    np.testing.assert_array_almost_equal(k.mass.gradient[0], grad1[0])

def test_gradient_spring():  
    np.testing.assert_array_almost_equal(k.spring.gradient[0], grad1[1])

def test_gradient_damper():  
    np.testing.assert_array_almost_equal(k.damper.gradient[0], grad1[2])

def test_gradient_scale():  
    np.testing.assert_array_almost_equal(k.scale.gradient, 2/grad1[3])

def test_gradient_sensitivity():  
    np.testing.assert_array_almost_equal(k.sensitivity.gradient[0], grad1[4])

