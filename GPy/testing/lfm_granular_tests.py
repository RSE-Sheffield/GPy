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
pregamma = baseline.get('pregamma').flatten()
preexp = baseline.get('preexp')

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
    result = lfmComputeH3(gamma, gamma, sigma2, X, X, preconst[1] - preconst[0])[0]
    np.testing.assert_array_almost_equal(result, baseline.get('baseline_computeH3'))

def test_lfmComputeH4():
    result = lfmComputeH4(gamma, gamma, sigma2, X, pregamma, preexp, mode = False, term = True)[0]
    np.testing.assert_array_almost_equal(result, baseline.get('baseline_computeH4'))

# ToDo:

# lfmGradientH31
# lfmGradientH32
# lfmGradientH41
# lfmGradientH42
# lfmGradientSigmaH3
# lfmGradientSigmaH4
