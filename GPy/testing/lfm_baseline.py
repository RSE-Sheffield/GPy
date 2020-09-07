# This file generates baseline data for testing functions associated with latent force models

from GPy.kern.src.lfm import *
import numpy as np
import sys

gamma1_p = 0.5 + 0.5j
sigma2 = 0.5
t1 = np.arange(4)
t2 = np.arange(3)

result_lfmUpsilonMatrix = lfmUpsilonMatrix(gamma1_p, sigma2, t1, t2)
print("result_lfmUpsilonMatrix")
print(result_lfmUpsilonMatrix)
np.savez('GPy/testing/baseline/result_lfmUpsilonMatrix.npz',result_lfmUpsilonMatrix)

result_lfmUpsilonVector = lfmUpsilonVector(gamma1_p, sigma2, t1)
print("result_lfmUpsilonVector")
print(result_lfmUpsilonVector)
np.savez('GPy/testing/baseline/result_lfmUpsilonVector.npz',result_lfmUpsilonVector)

result_lfmGradientUpsilonMatrix = lfmGradientUpsilonMatrix(gamma1_p, sigma2, t1, t2)
print("result_lfmGradientUpsilonMatrix")
print(result_lfmGradientUpsilonMatrix)
np.savez('GPy/testing/baseline/result_lfmGradientUpsilonMatrix.npz',result_lfmGradientUpsilonMatrix)

result_lfmGradientUpsilonVector = lfmGradientUpsilonVector(gamma1_p, sigma2, t1)
print("result_lfmGradientUpsilonVector")
print(result_lfmGradientUpsilonVector)
np.savez('GPy/testing/baseline/result_lfmGradientUpsilonVector.npz',result_lfmGradientUpsilonVector)

result_lfmGradientSigmaUpsilonMatrix = lfmGradientSigmaUpsilonMatrix(gamma1_p, sigma2, t1, t2)
print("result_lfmGradientSigmaUpsilonMatrix")
print(result_lfmGradientSigmaUpsilonMatrix)
np.savez('GPy/testing/baseline/result_lfmGradientSigmaUpsilonMatrix.npz',result_lfmGradientSigmaUpsilonMatrix)

result_lfmGradientSigmaUpsilonVector = lfmGradientSigmaUpsilonVector(gamma1_p, sigma2, t1)
print("result_lfmGradientSigmaUpsilonVector")
print(result_lfmGradientSigmaUpsilonVector)
np.savez('GPy/testing/baseline/result_lfmGradientSigmaUpsilonVector.npz',result_lfmGradientSigmaUpsilonVector)
