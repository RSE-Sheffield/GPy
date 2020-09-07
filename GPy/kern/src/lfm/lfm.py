# Copyright (c) 2012, James Hensman and Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# from .kern import Kern
# import numpy as np
# from ...core.parameterization import Param
# from paramz.transformations import Logexp
# from .independent_outputs import index_to_slices

from GPy.kern import Kern
import numpy as np
from GPy.core.parameterization import Param
from paramz.transformations import Logexp
from GPy.kern.src.independent_outputs import index_to_slices


from GPy.util.config import config # for assesing whether to use cython


class LFM(Kern):
    """
    Latent force model kernel toto: expand.
    """

    def __init__(self, input_dim, output_dim, scale=None, mass=None, spring=None, damper=None, sensitivity=None,
                 active_dims=None, isNormalised=None, name='lfm'):

        super(LFM, self).__init__(input_dim, active_dims, name)
        self.output_dim = output_dim

        if scale is None:
            scale = np.random.random()
        self.scale = Param('scale', scale, Logexp())

        if mass is None:
            mass = np.random.rand(self.output_dim)
        self.mass = Param('scale', mass, Logexp())

        if spring is None:
            spring = np.random.rand(self.output_dim)
        self.spring = Param('scale', spring, Logexp())

        if damper is None:
            damper = np.random.rand(self.output_dim)
        self.damper = Param('scale', damper, Logexp())

        if sensitivity is None:
            sensitivity = np.ones((self.output_dim, self.output_dim))
        self.sensitivity = sensitivity

        self.link_parameters(self.scale, self.mass, self.spring, self.damper, self.sensitivity)

        if isNormalised is None:
            isNormalised = [True for _ in range(self.output_dim)]
        self.isNormalised = isNormalised

        self.recalculate_intermediate_variables()

        # The kernel ALLWAYS puts the output index (the q for qth output) in the end of each rows.
        self.index_dim = -1

    def recalculate_intermediate_variables(self):

        # alpha and omega are intermediate variables used in the model and gradient for optimisation
        self.alpha = self.damper / (2 * self.mass)
        self.omega = np.sqrt(self.spring / self.mass - self.alpha * self.alpha)
        self.omega_isreal = np.isreal(self.omega)

    def parameters_changed(self):
        '''
        This function overrides the same name function in the grandparent class "Parameterizable", which is simply
        "pass"
        It describes the behaviours of the class when the "parameters" of a kernel are updated.
        '''
        self.recalculate_intermediate_variables()



    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        slices = index_to_slices(X[:, self.index_dim])
        slices2 = index_to_slices(X2[:, self.index_dim])
        target = np.zeros((X.shape[0], X2.shape[0]))
        for i in range(len(slices)):
            for j in range(len(slices2)):
                for k in range(len(slices[i])):
                    for l in range(len(slices2[j])):
                        K_sub_matrix = self.K_sub_matrix(i, X[slices[i][k], :], j, X2[slices2[j][l], :])
                        target.__setitem__((slices[i][k], slices2[j][l]), K_sub_matrix)
        return target


    def K_sub_matrix(self, q, X, q2= None, X2=None):
        # This K is a sub matrix as a part of K. It is covariance matrix between a pair of outputs.
        # The variable q and q2 are the index of outputs.
        # The variable X and X2 are subset of the X and X2 in K.
        if X2 is None:
            X2 = X
        if q2 is None:
            q2 = q
        assert (X.shape[2] == 1 or X.shape[2] == 1), 'Input can only have one column'

        # Creation  of the time matrices

        if self.omega_isreal[q] and self.omega_isreal[q2]:
            # Pre-computations to increase speed
            gamma1 = self.alpha[q] + 1j * self.omega[q]
            gamma2 = self.alpha[q2] + 1j * self.omega[q2]
            preGamma = np.array([gamma1 + gamma2,
                                 np.conj(gamma1) + gamma2
                                ])
            preConst = 1. / preGamma
            preExp1 = np.exp(-gamma1 * X)
            preExp2 = np.exp(-gamma2 * X2)
            # Actual computation  of the kernel
            sK = np.real(
                        self.lfmComputeH3(gamma1, gamma2, self.scale, X, X2, preConst, 0, 1)
                        +self.lfmComputeH3(gamma2, gamma1, self.scale, X2, X, preConst(2) - preConst(1), 0, 0).T
                        +self.lfmComputeH4(gamma1, gamma2, self.scale, X, preGamma, preExp2, 0, 1)
                        +self.lfmComputeH4(gamma2, gamma1, self.scale, X2, preGamma, preExp1, 0, 0).T
                        )
            if self.isNormalised[q]:
                K0 = (self.sensitivity[q] * self.sensitivity[q2]) / (
                        4 * np.sqrt(2) * self.mass[q] * self.mass[q2] * self.omega[q]*self.omega[q2])
            else:
                K0 = (np.sqrt(self.scale) * np.sqrt(np.pi) * self.sensitivity[q] * self.sensitivity[q2]) / (
                        4 * self.mass[q] * self.mass[q2] * self.omega[q]*self.omega[q2])

            K = K0 * sK
        else:
            # Pre-computations to increase the speed
            preExp1 = np.zeros(np.len(X), 2)
            preExp2 = np.zeros(np.len(X2), 2)
            gamma1_p = self.alpha[q]  + 1j * self.omega[q]
            gamma1_m = self.alpha[q]  - 1j * self.omega[q]
            gamma2_p = self.alpha[q2] + 1j * self.omega[q2]
            gamma2_m = self.alpha[q2] - 1j * self.omega[q2]
            preGamma = np.array([   gamma1_p + gamma2_p,
                                    gamma1_p + gamma2_m,
                                    gamma1_m + gamma2_p,
                                    gamma1_m + gamma2_m
                                ])
            preConst = 1. / preGamma
            preFactors = np.array([ preConst[2] - preConst[1],
                                    preConst[3] - preConst[4],
                                    preConst[3] - preConst[1],
                                    preConst[2] - preConst[4]
                                ])
            preExp1[:, 1] = np.exp(-gamma1_p * X)
            preExp1[:, 2] = np.exp(-gamma1_m * X)
            preExp2[:, 1] = np.exp(-gamma2_p * X2)
            preExp2[:, 2] = np.exp(-gamma2_m * X2)
            # Actual  computation of the kernel
            sK = (
                    self.lfmComputeH3(gamma1_p, gamma1_m, self.scale, X, X2, preFactors[1, 2], 1)
                    +self.lfmComputeH3(gamma2_p, gamma2_m, self.scale, X2, X, preFactors[3, 4], 1).T
                    +self.lfmComputeH4(gamma1_p, gamma1_m, self.scale, X, preGamma[1, 2, 4, 3], preExp2, 1)
                    +self.lfmComputeH4(gamma2_p, gamma2_m, self.scale, X2, preGamma[1, 3, 4, 2], preExp1, 1).T
                )
            if self.isNormalised[q]:
                K0 = (self.sensitivity[q] * self.sensitivity[q2]) / (
                        8 * np.sqrt(2) * self.mass[q] * self.mass [q2]*  self.omega[q]*self.omega[q2])
            else:
                K0 = (np.sqrt(self.scale) * np.sqrt(np.pi) * self.sensitivity[q] * self.sensitivity[q2]) / (
                        8 * self.mass[q] * self.mass[q2] *  self.omega[q]*self.omega[q2])

            K = K0 * sK
        return K

    def lfmComputeH3(self, gamma1_p, gamma1_m, sigma2, t1, t2, preFactor, mode=None, term=None):
        # LFMCOMPUTEH3 Helper function for computing part of the LFM kernel.
        # FORMAT
        # DESC computes a portion of the LFM kernel.
        # ARG gamma1 : Gamma value for first system.
        # ARG gamma2 : Gamma value for second system.
        # ARG sigma2 : length scale of latent process.
        # ARG t1 : first time input (number of time points x 1).
        # ARG t2 : second time input (number of time points x 1).
        # ARG mode: indicates in which way the vectors t1 and t2 must be transposed
        # RETURN h : result of this subcomponent of the kernel for the given values.

        if not mode:
            if not term:
                upsilon = lfmComputeUpsilonMatrix(gamma1_p, sigma2, t1, t2)
                h = preFactor * upsilon
            else:
                upsilon = lfmComputeUpsilonMatrix(gamma1_p, sigma2, t1, t2)
                h = -preFactor[0] * upsilon + preFactor[1] * np.conj(upsilon)

        else:
            upsilon = [lfmComputeUpsilonMatrix(gamma1_p, sigma2, t1, t2),
                          lfmComputeUpsilonMatrix(gamma1_m, sigma2, t1, t2)]
            h = preFactor[0]* upsilon[0] + preFactor[1] * upsilon[1]
        return h

    def lfmComputeH4(self, gamma1_p, gamma1_m, sigma2, t1, preFactor, preExp, mode=None, term=None ):
        # LFMCOMPUTEH4 Helper function for computing part of the LFM kernel.
        # FORMAT
        # DESC computes a portion of the LFM kernel.
        # ARG gamma1 : Gamma value for first system.
        # ARG gamma2 : Gamma value for second system.
        # ARG sigma2 : length scale of latent process.
        # ARG t1 : first time input (number of time points x 1).
        # ARG preFactor : precomputed constants
        # ARG preExp : precomputed exponentials
        # ARG mode: indicates in which way the vectors t1 and t2 must be transposed
        # RETURN h : result of this subcomponent of the kernel for the given values.

        if not mode:
            if not term:
                upsilon = lfmComputeUpsilonVector(gamma1_p, sigma2, t1)
                h = upsilon * (preExp / preFactor[0] - np.conj(preExp) / preFactor[1]).T

            else:
                upsilon= lfmComputeUpsilonVector(gamma1_p, sigma2, t1)
                h = upsilon * (preExp / preFactor[0]).T - np.conj(upsilon)*(preExp/preFactor[1]).T

        else:
            upsilon = [ lfmComputeUpsilonVector(gamma1_p, sigma2, t1),
                        lfmComputeUpsilonVector(gamma1_m, sigma2, t1) ]
            h = upsilon[0] * (preExp[:, 0] / preFactor[0] - preExp[:, 1] / preFactor[1]).T  + upsilon[1] * (preExp[:, 1] / preFactor[2] - preExp[:, 0] / preFactor[3]).T
        return h


    def Kdiag(self, X):
        return np.diag(self.B)[np.asarray(X, dtype=np.int).flatten()]

    def update_gradients_full(self, dL_dK, X, X2=None):
        index = np.asarray(X, dtype=np.int)
        if X2 is None:
            index2 = index
        else:
            index2 = np.asarray(X2, dtype=np.int)

        #attempt to use cython for a nasty double indexing loop: fall back to numpy
        if use_coregionalize_cython:
            dL_dK_small = self._gradient_reduce_cython(dL_dK, index, index2)
        else:
            dL_dK_small = self._gradient_reduce_numpy(dL_dK, index, index2)


        dkappa = np.diag(dL_dK_small).copy()
        dL_dK_small += dL_dK_small.T
        dW = (self.W[:, None, :]*dL_dK_small[:, :, None]).sum(0)

        self.W.gradient = dW
        self.kappa.gradient = dkappa

    def _gradient_reduce_numpy(self, dL_dK, index, index2):
        index, index2 = index[:,0], index2[:,0]
        dL_dK_small = np.zeros_like(self.B)
        for i in range(self.output_dim):
            tmp1 = dL_dK[index==i]
            for j in range(self.output_dim):
                dL_dK_small[j,i] = tmp1[:,index2==j].sum()
        return dL_dK_small


    def update_gradients_diag(self, dL_dKdiag, X):
        index = np.asarray(X, dtype=np.int).flatten()
        dL_dKdiag_small = np.array([dL_dKdiag[index==i].sum() for i in range(self.output_dim)])
        self.W.gradient = 2.*self.W*dL_dKdiag_small[:, None]
        self.kappa.gradient = dL_dKdiag_small

    def gradients_X(self, dL_dK, X, X2=None):
        return np.zeros(X.shape)

    def gradients_X_diag(self, dL_dKdiag, X):
        return np.zeros(X.shape)
