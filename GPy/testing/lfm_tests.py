# This is *very* work in progress!

import itertools

import GPy
from GPy import kern

# my_lfm = kern.LFM(input_dim = 1 , output_dim = 1)

# print(my_lfm)

k_lfmxlfm = [kern.LFMXLFM(input_dim = 1 , output_dim = 1) for i in range(9)]

# print(my_lfmxlfm)

cov_dict = {(0,0): k_lfmxlfm[0],
            (0,1): k_lfmxlfm[1],
            (0,2): k_lfmxlfm[2],
            (1,0): k_lfmxlfm[3],
            (1,1): k_lfmxlfm[4],
            (1,2): k_lfmxlfm[5],
            (2,0): k_lfmxlfm[6],
            (2,1): k_lfmxlfm[7],
            (2,2): k_lfmxlfm[8]}

k = GPy.kern.MultioutputKern(k_lfmxlfm, cross_covariances=cov_dict)
#k = GPy.kern.MultioutputKern([k1, k2])