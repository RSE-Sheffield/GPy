import lfm_C
import numpy as np
import sys

def lfmUpsilonMatrix(gamma1_p, sigma2, t1, t2):
    return lfm_C.UpsilonMatrix(gamma1_p, sigma2, t1.astype(np.float64), t2.astype(np.float64))

def lfmUpsilonVector(gamma1_p, sigma2, t1):
    return lfm_C.UpsilonVector(gamma1_p, sigma2, t1.astype(np.float64))

def lfmGradientUpsilonMatrix(gamma1_p, sigma2, t1, t2):
    return lfm_C.GradientUpsilonMatrix(gamma1_p, sigma2, t1.astype(np.float64), t2.astype(np.float64))

def lfmGradientUpsilonVector(gamma1_p, sigma2, t1):
    return lfm_C.GradientUpsilonVector(gamma1_p, sigma2, t1.astype(np.float64))

def lfmGradientSigmaUpsilonMatrix(gamma1_p, sigma2, t1, t2):
    return lfm_C.GradientSigmaUpsilonMatrix(gamma1_p, sigma2, t1.astype(np.float64), t2.astype(np.float64))

def lfmGradientSigmaUpsilonVector(gamma1_p, sigma2, t1):
    return lfm_C.GradientSigmaUpsilonVector(gamma1_p, sigma2, t1.astype(np.float64))

if __name__ == '__main__':
    gamma1_p = 0.5 + 0.5j
    sigma2 =0.5
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
