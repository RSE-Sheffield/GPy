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

result_lfmUpsilonVector = lfmUpsilonVector(gamma1_p, sigma2, t1)
print("result_lfmUpsilonVector")
print(result_lfmUpsilonVector)

result_lfmGradientUpsilonMatrix = lfmGradientUpsilonMatrix(gamma1_p, sigma2, t1, t2)
print("result_lfmGradientUpsilonMatrix")
print(result_lfmGradientUpsilonMatrix)

result_lfmGradientUpsilonVector = lfmGradientUpsilonVector(gamma1_p, sigma2, t1)
print("result_lfmGradientUpsilonVector")
print(result_lfmGradientUpsilonVector)

result_lfmGradientSigmaUpsilonMatrix = lfmGradientSigmaUpsilonMatrix(gamma1_p, sigma2, t1, t2)
print("result_lfmGradientSigmaUpsilonMatrix")
print(result_lfmGradientSigmaUpsilonMatrix)

result_lfmGradientSigmaUpsilonVector = lfmGradientSigmaUpsilonVector(gamma1_p, sigma2, t1)
print("result_lfmGradientSigmaUpsilonVector")
print(result_lfmGradientSigmaUpsilonVector)
