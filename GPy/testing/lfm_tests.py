from GPy.kern.src.lfm import *
import numpy as np
import sys
import unittest

gamma1_p = 0.5 + 0.5j
sigma2 = 0.5
t1 = np.arange(4)
t2 = np.arange(3)

def test_lfmUpsilonMatrix():
    result = lfmUpsilonMatrix(gamma1_p, sigma2, t1, t2)
    baseline = np.load('GPy/testing/baseline/result_lfmUpsilonMatrix.npz')['arr_0']
    np.testing.assert_array_almost_equal(result, baseline)

def test_lfmUpsilonVector():
    result = lfmUpsilonVector(gamma1_p, sigma2, t1)
    baseline = np.load('GPy/testing/baseline/result_lfmUpsilonVector.npz')['arr_0']
    np.testing.assert_array_almost_equal(result, baseline)

def test_lfmGradientUpsilonMatrix():
    result = lfmGradientUpsilonMatrix(gamma1_p, sigma2, t1, t2)
    baseline = np.load('GPy/testing/baseline/result_lfmGradientUpsilonMatrix.npz')['arr_0']
    np.testing.assert_array_almost_equal(result, baseline)

def test_lfmGradientUpsilonVector():
    result = lfmGradientUpsilonVector(gamma1_p, sigma2, t1)
    baseline = np.load('GPy/testing/baseline/result_lfmGradientUpsilonVector.npz')['arr_0']
    np.testing.assert_array_almost_equal(result, baseline)

def test_lfmGradientSigmaUpsilonMatrix():
    result = lfmGradientSigmaUpsilonMatrix(gamma1_p, sigma2, t1, t2)
    baseline = np.load('GPy/testing/baseline/result_lfmGradientSigmaUpsilonMatrix.npz')['arr_0']
    np.testing.assert_array_almost_equal(result, baseline)

def test_lfmGradientSigmaUpsilonVector():
    result = lfmGradientSigmaUpsilonVector(gamma1_p, sigma2, t1)
    baseline = np.load('GPy/testing/baseline/result_lfmGradientSigmaUpsilonVector.npz')['arr_0']
    np.testing.assert_array_almost_equal(result, baseline)

