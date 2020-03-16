#import GPy.kern.src.lfm
from GPy.kern.src.lfm import lfm_C
import numpy as np
import sys

def lfmcomputeUpsilonMatrix(gamma1_p, sigma2, t1, t2):
    return lfm_C.UpsilonMatrix(gamma1_p, sigma2, t1, t2)

if __name__ == '__main__':
    gamma1_p = 0.5 + 0.5j
    sigma2 = 0.5
    t1 = np.arange(4)
    t2 = np.arange(3)
    result = lfmcomputeUpsilonMatrix(gamma1_p, sigma2, t1, t2)
    print(result)