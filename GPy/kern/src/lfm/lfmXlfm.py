from GPy.kern import Kern
from GPy.kern.src.lfm import *
import numpy as np
from GPy.core.parameterization import Param
from paramz.transformations import Logexp
from GPy.kern.src.independent_outputs import index_to_slices


from GPy.util.config import config # for assesing whether to use cython

class LFMXLFM(Kern):
    """
    LFM X LFM convolved kernel todo: expand.
    """

    def __init__(self, input_dim, scale=None, mass=None, spring=None, damper=None, sensitivity=None,
                 active_dims=None, isNormalised=None, name='lfmXlfm'):

        super(LFMXLFM, self).__init__(input_dim, active_dims, name)
        # Parameters go in as vectors of length 2, one parameter per kernel in the cross

        if scale is None:
            scale =  np.ones(1) # np.random.rand(self.output_dim)
        self.scale = Param('scale', scale)

        if mass is None:
            mass =  np.ones(2) # np.random.rand(self.output_dim)
        self.mass = Param('mass', mass)

        if spring is None:
            spring =  np.ones(2) # np.random.rand(self.output_dim)
        self.spring = Param('spring', spring)

        if damper is None:
            damper = np.ones(2) # np.random.rand(self.output_dim)
        self.damper = Param('damper', damper)

        if sensitivity is None:
            sensitivity = np.ones(2) #np.random.rand(self.output_dim) # np.ones((self.output_dim, self.input_dim))
        self.sensitivity = Param('sensitivity', sensitivity)

        self.link_parameters(self.scale, self.mass, self.spring, self.damper, self.sensitivity)

        if isNormalised is None:
            isNormalised = [True for _ in range(2)]
        self.isNormalised = isNormalised

        self.recalculate_intermediate_variables()

        # The kernel ALLWAYS puts the output index (the q for qth output) in the end of each rows.
        self.index_dim = -1



    def recalculate_intermediate_variables(self):
        # Get length scale out.
        self.sigma2 = self.scale[0] #in gpmat, the first kernel's scale is used and the scond apparently ignored
        self.sigma = np.sqrt(self.sigma2) #assuming this is a good sub for `csqrt`
        # alpha and omega are intermediate variables used in the model and gradient for optimisation
        self.alpha = self.damper / (2 * self.mass)
        self.omega = np.sqrt(self.spring / self.mass - self.alpha * self.alpha)
        self.omega_isreal = np.isreal(self.omega).all()

        self.gamma = self.alpha + 1j * self.omega

    def parameters_changed(self):
        '''
        This function overrides the same name function in the grandparent class "Parameterizable", which is simply
        "pass"
        It describes the behaviours of the class when the "parameters" of a kernel are updated.
        '''
        self.recalculate_intermediate_variables()
        # super(LFM, self).parameters_changed()



    def K(self, X, X2=None):
        if X2 is None:
            X2 = X

        # Creation  of the time matrices

        if self.omega_isreal:
            # Pre-computations to increase speed
            gamma1 = self.alpha[0] + 1j * self.omega[0]
            gamma2 = self.alpha[1] + 1j * self.omega[1]
            # print('gamma1')
            # print(gamma1)
            # print('gamma2')
            # print(gamma2)
            preGamma = np.array([gamma1 + gamma2,
                                 np.conj(gamma1) + gamma2
                                ])
            preConsX = 1. / preGamma
            preExp1 = np.exp(-gamma1 * X)
            preExp2 = np.exp(-gamma2 * X2)
            # Actual computation  of the kernel
            sK = np.real(
                        lfmComputeH3(gamma1, gamma2, self.sigma, X, X2, preConsX, 0, 1)[0]
                        + lfmComputeH3(gamma2, gamma1, self.sigma, X2, X, preConsX[1] - preConsX[0], 0, 0)[0].T
                        + lfmComputeH4(gamma1, gamma2, self.sigma, X, preGamma, preExp2, 0, 1)[0]
                        + lfmComputeH4(gamma2, gamma1, self.sigma, X2, preGamma, preExp1, 0, 0)[0].T
                        )
            if self.isNormalised:
                K0 = (self.sensitivity[0] * self.sensitivity[1]) / (
                        4 * np.sqrt(2) * self.mass[0] * self.mass[1] * self.omega[0] * self.omega[1])
            else:
                K0 = (np.sqrt(self.sigma) * np.sqrt(np.pi) * self.sensitivity[0] * self.sensitivity[1]) / (
                        4 * self.mass[0] * self.mass[1] * self.omega[0] * self.omega[1])

            K = K0 * sK
        else:
            # Pre-computations to increase the speed
            preExp1 = np.zeros((np.max(np.shape(X)), 2))
            preExp2 = np.zeros((np.max(np.shape(X2)), 2))
            gamma1_p = self.alpha  + 1j * self.omega
            gamma1_m = self.alpha  - 1j * self.omega
            gamma2_p = self.alpha + 1j * self.omega
            gamma2_m = self.alpha - 1j * self.omega
            preGamma = np.array([   gamma1_p + gamma2_p,
                                    gamma1_p + gamma2_m,
                                    gamma1_m + gamma2_p,
                                    gamma1_m + gamma2_m
                                ])
            preConsX = 1. / preGamma
            preFactors = np.array([ preConsX[1] - preConsX[0],
                                    preConsX[2] - preConsX[3],
                                    preConsX[2] - preConsX[0],
                                    preConsX[1] - preConsX[3]
                                ])
            preExp1[:, 0] = np.exp(-gamma1_p * X).ravel()
            preExp1[:, 1] = np.exp(-gamma1_m * X).ravel()
            preExp2[:, 0] = np.exp(-gamma2_p * X2).ravel()
            preExp2[:, 1] = np.exp(-gamma2_m * X2).ravel()
            # Actual  computation of the kernel
            sK = (
                    lfmComputeH3(gamma1_p, gamma1_m, self.sigma, X, X2, preFactors[np.array([0, 1])], 1)[0]
                    + lfmComputeH3(gamma2_p, gamma2_m, self.sigma, X2, X, preFactors[np.array([2, 3])], 1)[0].T
                    + lfmComputeH4(gamma1_p, gamma1_m, self.sigma, X, preGamma[np.array([0, 1, 3, 2])], preExp2, 1)[0]
                    + lfmComputeH4(gamma2_p, gamma2_m, self.sigma, X2, preGamma[np.array([0, 2, 3, 1])], preExp1, 1)[0].T
                )
            if self.isNormalised:
                K0 = (self.sensitivity[0] * self.sensitivity[1]) / (
                        8 * np.sqrt(2) * self.mass[0] * self.mass[1] *  self.omega[0] * self.omega[1])
            else:
                K0 = (self.sigma * np.sqrt(np.pi) * self.sensitivity[0] * self.sensitivity[1]) / (
                        8 * self.mass[0] * self.mass[1] * self.omega[0] * self.omega[1])

            K = K0 * sK
        return K

    def Kdiag(self, X):
        assert X.shape[1] == 2, 'Input can only have one column'
        slices = index_to_slices(X[:,self.index_dim])
        target = np.zeros((X.shape[0]))  #.astype(np.complex128)
        for q, slices_i in zip(range(2), slices):
            for s in slices_i:
                Kdiag_sub = np.real(self.Kdiag_sub(q, X[s, :-1]))
                np.copyto(target[s], Kdiag_sub)

        return target

    def Kdiag_sub(self, q, X):

        def lfmDiagComputeH3(gamma, sigma2, t, factor, preExp, mode):
            if mode:
                vec = np.multiply(preExp, lfmUpsilonVector(gamma, sigma2, t)) * factor
            else:
                temp = np.multiply(preExp, lfmUpsilonVector(gamma, sigma2, t))
                vec = 2 * np.real(temp / factor[0]) - temp / factor[1]
            return vec

        def lfmDiagComputeH4(gamma, sigma2, t, factor, preExp, mode):
            if mode:
                vec = (preExp[:, 0] / factor[0] - 2 * preExp[:, 1] / factor[1]) * lfmUpsilonVector(gamma, sigma2, t)
            else:
                temp = lfmUpsilonVector(gamma, sigma2, t)
                temp2 =temp * np.conj(preExp) / factor[1]
                vec = (temp*preExp) / factor[0] - 2 * np.real(temp2)
            return vec

       # preExp = np.zeros((len(X), 2)).astype(np.complex128)
        gamma_p = self.alpha[q] + 1j * self.omega[q]
        gamma_m = self.alpha[q] - 1j * self.omega[q]
        preFactors = np.array([2 / (gamma_p + gamma_m) - 1 / gamma_m,
                               2 / (gamma_p + gamma_m) - 1 / gamma_p])
        preExp = np.hstack([np.exp(-gamma_p * X), np.exp(-gamma_m * X)])
        sigma2 = self.sigma2
        # Actual computation of the kernel
        sk = lfmDiagComputeH3(-gamma_m, sigma2, X, preFactors[0], preExp[:, 1], 1)  \
            + lfmDiagComputeH3(-gamma_p, sigma2, X, preFactors[1], preExp[:, 0], 1)  \
            + lfmDiagComputeH4(gamma_m, sigma2, X, [gamma_m, (gamma_p + gamma_m)], np.hstack([preExp[:, 1][:,None], preExp[:, 0][:,None]]), 1)  \
            + lfmDiagComputeH4(gamma_p, sigma2, X, [gamma_p, (gamma_p + gamma_m)], preExp, 1)
        if self.isNormalised:
            k0 = self.sensitivity[q] ** 2 / (8 * np.sqrt(2) * self.mass[q] ** 2 * self.omega[q] ** 2)
        else:
            k0 = np.sqrt(np.pi) * self.sigma[q] * self.sensitivity[q] ** 2 / (8 * self.mass[q] ** 2 * self.omega[q] ** 2)
        k = np.dot(k0, k0) * sk
        return k



    def update_gradients_full(self, dL_dK, X, X2=None, meanVector=None):
        """
        Given the derivative of the objective wrt the covariance matrix
        (dL_dK), compute the gradient wrt the parameters of this kernel,
        and store in the parameters object as e.g. self.variance.gradient
        """
        # FORMAT
        # DESC computes cross kernel terms between two LFM kernels for
        # the multiple output kernel.
        # ARG lfmKern1 : the kernel structure associated with the first LFM
        # kernel.
        # ARG lfmKern2 : the kernel structure associated with the second LFM
        # kernel.
        # ARG X : row inputs for which kernel is to be computed.
        # ARG t2 : column inputs for which kernel is to be computed.
        # ARG covGrad : gradient of the objective function with respect to
        # the elements of the cross kernel matrix.
        # ARG meanVec : precomputed factor that is used for the switching dynamical
        # latent force model.
        # RETURN g1 : gradient of the parameters of the first kernel, for
        # ordering see lfmKernExtractParam.
        # RETURN g2 : gradient of the parameters of the second kernel, for
        # ordering see lfmKernExtractParam.
        #
        # SEEALSO : multiKernParamInit, multiKernCompute, lfmKernParamInit, lfmKernExtractParam
        #
        # COPYRIGHT : David Luengo, 2007, 2008, Mauricio Alvarez, 2008
        #
        # MODIFICATIONS : Neil D. Lawrence, 2007
        #
        # MODIFICATIONS : Mauricio A. Alvarez, 2010

        # KERN

        if X2 is None: X2 = X

        subComponent = False  # This is just a flag that indicates if this kernel is part of a bigger kernel (SDLFM)
        covGrad = dL_dK
        if covGrad is None and meanVector is None:
            covGrad = X2
            X2 = X
        elif covGrad is not None and  meanVector is not None:
            subComponent = True
            if np.size(meanVector) > 1:
                if np.shape(meanVector, 1) == 1:
                    assert np.shape(meanVector, 2)==np.shape(covGrad, 2), 'The dimensions of meanVector don''t correspond to the dimensions of covGrad'
                else:
                    assert np.shape(meanVector.conj().T, 2)==np.shape(covGrad,2), 'The dimensions of meanVector don''t correspond to the dimensions of covGrad'
            else:
                if np.size(X) == 1 and np.size(X2) > 1:
                    # matGrad will be row vector and so should be covGrad
                    dimcovGrad = np.max(np.shape((covGrad)))
                    covGrad = covGrad.reshape(1, dimcovGrad,order='F').copy()
                elif np.size(X) > 1 and np.size(X2) == 1:
                    # matGrad will be column vector and sp should be covGrad
                    dimcovGrad = np.shape.max(covGrad)
                    covGrad = covGrad.reshape(dimcovGrad,1,order='F').copy()

        # assert np.shape(X)[1] == 1 or np.shape(X2)[1] == 1,  'Input can only have one column. np.shape(X) = ' + str(np.shape(X)) + 'np.shape(X2) =' + str(np.shape(X2))
        # assert  self.scale[q] == self.scale[q2], '''Kernels cannot be cross combined if they have different inverse 
        #                                          widths. self.scale[%d] = %.5f = self.scale[%d] = %.5f''' % (q, q2, self.scale[q], self.scale[q2])

        # Parameters of the simulation (in the order provided by kernExtractParam in the matlab code)
        
        m = self.mass.values # Par. 1
        D = self.spring.values  # Par. 2
        C = self.damper.values # Par. 3
        sigma2 = self.scale.values[0] # Par. 4
        sigma = np.sqrt(sigma2)
        S = self.sensitivity.values # Par. 5

        alpha = C / (2 * m)
        omega = np.sqrt(D / m - alpha ** 2)

        # Initialization of vectors and matrices

        g1 = np.zeros((4 + 2))
        g2 = np.zeros((4 + 2))

        # Precomputations

        if np.all(np.isreal(omega)):
            computeH = cell(4, 1)
            computeUpsilonMatrix = cell(2, 1)
            computeUpsilonVector = cell(2, 1)
            gradientUpsilonMatrix = cell(2, 1)
            gradientUpsilonVector = cell(2, 1)
            gamma1 = alpha[0] + 1j * omega[0]
            gamma2 = alpha[1] + 1j * omega[1]
            gradientUpsilonMatrix[0]= lfmGradientUpsilonMatrix(gamma1, sigma2, X, X2)
            gradientUpsilonMatrix[1]= lfmGradientUpsilonMatrix(gamma2, sigma2, X2, X)
            gradientUpsilonVector[0]= lfmGradientUpsilonVector(gamma1, sigma2, X)
            gradientUpsilonVector[1]= lfmGradientUpsilonVector(gamma2, sigma2, X2)
            preGamma= np.array([gamma1 + gamma2, np.conj(gamma1) + gamma2])
            preGamma2 = np.power(preGamma, 2)
            preConsX = 1 / preGamma
            preConsX2 = 1 / preGamma2
            preExp1 = np.exp(-gamma1 * X)
            preExp2 = np.exp(-gamma2 * X2)
            preExpX = np.multiply(X,np.exp(-gamma1 * X))
            preExpX2 = np.multiply(X2, np.exp(-gamma2 * X2))
            [computeH[0], computeUpsilonMatrix[0]] = lfmComputeH3(gamma1, gamma2, sigma2, X, X2, preConsX, 0, 1)
            [computeH[1], computeUpsilonMatrix[1]] = lfmComputeH3(gamma2, gamma1, sigma2, X2, X, preConsX[1] - preConsX[0], 0, 0)
            [computeH[2], computeUpsilonVector[0]] = lfmComputeH4(gamma1, gamma2, sigma2, X, preGamma, preExp2, 0, 1  )
            [computeH[3], computeUpsilonVector[1]] = lfmComputeH4(gamma2, gamma1, sigma2, X2, preGamma, preExp1, 0, 0 )
            preKernel = np.real( computeH[0]+ computeH[1].T + computeH[2]+ computeH[3].T)
        else:
            computeH  = cell(4, 1)
            computeUpsilonMatrix = cell(2, 1)
            computeUpsilonVector = cell(2, 1)
            gradientUpsilonMatrix = cell(4, 1)
            gradientUpsilonVector = cell(4, 1)
            gamma1_p = alpha[0] + 1j * omega[0]
            gamma1_m = alpha[0] - 1j * omega[0]
            gamma2_p = alpha[1] + 1j * omega[1]
            gamma2_m = alpha[1] - 1j * omega[1]
            gradientUpsilonMatrix[0]= lfmGradientUpsilonMatrix(gamma1_p, sigma2, X, X2)
            gradientUpsilonMatrix[1]= lfmGradientUpsilonMatrix(gamma1_m, sigma2, X, X2)
            gradientUpsilonMatrix[2]= lfmGradientUpsilonMatrix(gamma2_p, sigma2, X2, X)
            gradientUpsilonMatrix[3]= lfmGradientUpsilonMatrix(gamma2_m, sigma2, X2, X)
            gradientUpsilonVector[0]= lfmGradientUpsilonVector(gamma1_p, sigma2, X)
            gradientUpsilonVector[1]= lfmGradientUpsilonVector(gamma1_m, sigma2, X)
            gradientUpsilonVector[2]= lfmGradientUpsilonVector(gamma2_p, sigma2, X2)
            gradientUpsilonVector[3]= lfmGradientUpsilonVector(gamma2_m, sigma2, X2)
            preExp1 = np.zeros((np.max(np.shape(X), 2)))
            preExp2 = np.zeros((np.max(np.shape(X2), 2)))
            preExpX = np.zeros((np.max(np.shape(X), 2)))
            preExpX2 = np.zeros((np.max(np.shape(X2), 2)))
            preGamma = np.array([gamma1_p + gamma2_p,
                                   gamma1_p + gamma2_m,
                                   gamma1_m + gamma2_p,
                                   gamma1_m + gamma2_m])
            preGamma2 = np.power(preGamma, 2)
            preConsX = 1 / preGamma
            preConsX2 = 1 / preGamma2
            preFactors = np.array([preConsX[1] - preConsX[0],
                                   preConsX[2] - preConsX[3],
                                   preConsX[2] - preConsX[0],
                                   preConsX[1] - preConsX[3]])
            preFactors2 = np.array([-preConsX2[1] + preConsX2[0],
                                    -preConsX2[2] + preConsX2[3],
                                    -preConsX2[2] + preConsX2[0],
                                    -preConsX2[1] + preConsX2[3]])
            preExp1[:, 0] = np.exp(-gamma1_p * X)
            preExp1[:, 1] = np.exp(-gamma1_m * X)
            preExp2[:, 0] = np.exp(-gamma2_p * X2)
            preExp2[:, 1] = np.exp(-gamma2_m * X2)
            preExpX[:, 0] = X * np.exp(-gamma1_p * X)
            preExpX[:, 1] = X * np.exp(-gamma1_m * X)
            preExpX2[:, 0] = X2 * np.exp(-gamma2_p * X2)
            preExpX2[:, 1] = X2 * np.exp(-gamma2_m * X2)
            [computeH[0], computeUpsilonMatrix[0]] = lfmComputeH3(gamma1_p, gamma1_m, sigma2, X, X2, preFactors[1, 2], 1)
            [computeH[1], computeUpsilonMatrix[1]] = lfmComputeH3(gamma2_p, gamma2_m, sigma2, X2, X, preFactors[3, 4], 1)
            [computeH[2], computeUpsilonVector[0]] = lfmComputeH4(gamma1_p, gamma1_m, sigma2, X, preGamma[1, 2, 4, 3], preExp2, 1)
            [computeH[3], computeUpsilonVector[1]] = lfmComputeH4(gamma2_p, gamma2_m, sigma2, X2, preGamma[1, 3, 4, 2], preExp1, 1)
            preKernel = computeH[0]+ computeH[1].T + computeH[2]+ computeH[3].T

        if np.all(np.isreal(omega)):
            if self.isNormalised[0]:
                K0 = np.prod(S) / (4 * np.sqrt(2) * np.prod(m) * np.prod(omega))
            else:
                K0 = sigma * np.prod(S) * np.sqrt(np.pi) / (4 * np.prod(m) * np.prod(omega))
        else:
            if self.isNormalised[1]:
                K0 = (np.prod(S) / (8 * np.sqrt(2) * np.prod(m) * np.prod(omega)))
            else:
                K0 = (sigma * np.prod(S) * np.sqrt(np.pi) / (8 * np.prod(m) * np.prod(omega)))

        # Gradient with respect to m, D and C
        for ind_theta in np.arange(3):  # Parameter (m, D or C)
            for ind_par in np.arange(2):  # System (1 or 2)
                # Choosing the right gradients for m, omega, gamma1 and gamma2
                if ind_theta == 0: # Gradient wrt m
                    gradThetaM = [1 - ind_par, ind_par]
                    gradThetaAlpha = -C / (2 * np.power(m, 2))
                    gradThetaOmega = (np.power(C,2) - 2 * m * D) / (2 * (m ** 2) * np.sqrt(4 * m * D - C ** 2))
                if ind_theta == 1: # Gradient wrt D
                    gradThetaM = np.zeros((2))
                    gradThetaAlpha = np.zeros((2))
                    gradThetaOmega = 1 / np.sqrt(4 * m * D - np.power(C, 2))
                if ind_theta == 2:  # Gradient wrt C
                    gradThetaM = np.zeros((2))
                    gradThetaAlpha = 1 / (2 * m)
                    gradThetaOmega = -C / (2 * m * np.sqrt(4 * m * D - C ** 2))

                gradThetaGamma1 = gradThetaAlpha + 1j * gradThetaOmega
                gradThetaGamma2 = gradThetaAlpha - 1j * gradThetaOmega

                # Gradient evaluation
                if np.all(np.isreal(omega)):
                    gradThetaGamma2 = gradThetaGamma1[1]
                    gradThetaGamma1 = [gradThetaGamma1[0], np.conj(gradThetaGamma1[0])]

                    #  gradThetaGamma1 =  gradThetaGamma11
                    if not ind_par:
                        matGrad = K0 * \
                                  np.real(lfmGradientH31(preConsX, preConsX2, gradThetaGamma1,
                                                         gradientUpsilonMatrix[0], 1, computeUpsilonMatrix[0], 1, 0, 1)
                                          + lfmGradientH32(preGamma2, gradThetaGamma1, computeUpsilonMatrix[1],
                                                           1, 0, 0).T
                                          + lfmGradientH41(preGamma, preGamma2, gradThetaGamma1, preExp2,
                                                           gradientUpsilonVector[0], 1, computeUpsilonVector[0], 1, 0, 1)
                                          + lfmGradientH42(preGamma, preGamma2, gradThetaGamma1, preExp1, preExpX,
                                                           computeUpsilonVector[1], 1, 0, 0).T \
                                          - (gradThetaM[ind_par] / m[ind_par]
                                             + gradThetaOmega[ind_par] / omega[ind_par]) * preKernel)
                    else:
                        matGrad = K0 * \
                                  np.real(lfmGradientH31((preConsX[1] - preConsX[0]), (-preConsX2[1] + preConsX2[0]),
                                                         gradThetaGamma2, gradientUpsilonMatrix[1], 1,
                                                         computeUpsilonMatrix[1], 1, 0, 0).T
                                          + lfmGradientH32(preConsX2, gradThetaGamma2, computeUpsilonMatrix[0],1, 0, 1)
                                          + lfmGradientH41(preGamma, preGamma2, gradThetaGamma2, preExp1,
                                                           gradientUpsilonVector[1], 1, computeUpsilonVector[1], 1, 0, 0).T
                                          + lfmGradientH42(preGamma, preGamma2, gradThetaGamma2, preExp2, preExpX2,
                                                           computeUpsilonVector[0], 1, 0, 1) \
                                          - (gradThetaM[ind_par] / m[ind_par] + gradThetaOmega[ind_par] / omega[ind_par])
                                            * preKernel)

                else:

                    gradThetaGamma11 = np.array([gradThetaGamma1[0], gradThetaGamma2[0]])
                    gradThetaGamma2 = np.array([gradThetaGamma1[1], gradThetaGamma2[1]])
                    gradThetaGamma1 = gradThetaGamma11

                    if not ind_par:  # ind_par = k
                        matGrad = K0 *  \
                                    (lfmGradientH31(preFactors[np.array([0,1])], preFactors2[np.array([0,1])], gradThetaGamma1, gradientUpsilonMatrix[0],    # preFactors[0,1], preFactors2[0,1], gradThetaGamma1, gradientUpsilonMatrix[0],
                                                    gradientUpsilonMatrix[1], computeUpsilonMatrix[0][0], computeUpsilonMatrix[0][1], 1)
                                    + lfmGradientH32(preGamma2, gradThetaGamma1, computeUpsilonMatrix[1][0], computeUpsilonMatrix[1][1], 1).T
                                    + lfmGradientH41(preGamma, preGamma2, gradThetaGamma1, preExp2, gradientUpsilonVector[0],
                                                     gradientUpsilonVector[1], computeUpsilonVector[0][0], computeUpsilonVector[0][1], 1)
                                    + lfmGradientH42(preGamma, preGamma2, gradThetaGamma1, preExp1, preExpX, computeUpsilonVector[1][0],
                                                     computeUpsilonVector[1][1], 1).T
                                    - (gradThetaM[ind_par] / m[ind_par]+ gradThetaOmega[ind_par] / omega[ind_par]) * preKernel)

                    else:  # ind_par = r
                        matGrad = K0 * \
                                  (lfmGradientH31(preFactors[2,3], preFactors2[2,3], gradThetaGamma2,
                                                  gradientUpsilonMatrix[2], gradientUpsilonMatrix[3],
                                                  computeUpsilonMatrix[1][0], computeUpsilonMatrix[1][1], 1).T
                                   + lfmGradientH32(preGamma2([0, 2, 1, 3]), gradThetaGamma2, computeUpsilonMatrix[0][0],
                                                    computeUpsilonMatrix[0][1], 1)
                                   + lfmGradientH41(preGamma[0,2,1,3], preGamma2[0,2,1,3], gradThetaGamma2, preExp1,
                                                    gradientUpsilonVector[2], gradientUpsilonVector[3],
                                                    computeUpsilonVector[1][0],computeUpsilonVector[1][1], 1).T
                                   + lfmGradientH42(preGamma[0,2,1,3], preGamma2[0,2,1,3], gradThetaGamma2, preExp2,
                                                    preExpX2, computeUpsilonVector[0][0], computeUpsilonVector[0][1], 1)
                                   - (gradThetaM[ind_par] / m[ind_par] + gradThetaOmega[ind_par] / omega[ind_par])
                                     * preKernel)

                if subComponent:
                    if np.shape(meanVector)[0] == 1:
                        matGrad = matGrad * meanVector
                    else:
                        matGrad = (meanVector * matGrad).T
                # Check the parameter to assign the derivative
                if ind_par == 0:
                    g1[ind_theta] = sum(sum(matGrad * covGrad))
                else:
                    g2[ind_theta] = sum(sum(matGrad * covGrad))


        # Gradients with respect to sigma

        if np.all(np.isreal(omega)):
            if self.isNormalised[0]:
                matGrad = K0 * \
                          np.real(lfmGradientSigmaH3(gamma1, gamma2, sigma2, X, X2, preConsX, 0, 1)\
                                  + lfmGradientSigmaH3(gamma2, gamma1, sigma2, X2, X, preConsX[1] - preConsX[0], 0, 0).T\
                                  + lfmGradientSigmaH4(gamma1, gamma2, sigma2, X, preGamma, preExp2, 0, 1  )\
                                  + lfmGradientSigmaH4(gamma2, gamma1, sigma2, X2, preGamma, preExp1, 0, 0 ).T)
            else:
                matGrad = (np.prod(S) * np.sqrt(np.pi) / (4 * np.prod(m) * np.prod(omega))) \
                          * np.real(preKernel
                                    + sigma
                                      * (lfmGradientSigmaH3(gamma1, gamma2, sigma2, X, X2, preConsX, 0, 1)
                                        +  lfmGradientSigmaH3(gamma2, gamma1, sigma2, X2, X, preConsX[1] - preConsX[0], 0, 0).T
                                        +  lfmGradientSigmaH4(gamma1, gamma2, sigma2, X, preGamma, preExp2, 0, 1  )
                                        +  lfmGradientSigmaH4(gamma2, gamma1, sigma2, X2, preGamma, preExp1, 0, 0 ).T))
        else:
            if self.isNormalised[0]:
                matGrad = K0 * \
                          (lfmGradientSigmaH3(gamma1_p, gamma1_m, sigma2, X, X2, preFactors[0,1], 1)\
                           +  lfmGradientSigmaH3(gamma2_p, gamma2_m, sigma2, X2, X, preFactors[2,3], 1).T\
                           +  lfmGradientSigmaH4(gamma1_p, gamma1_m, sigma2, X, preGamma[0,1,3,2], preExp2, 1 )\
                           +  lfmGradientSigmaH4(gamma2_p, gamma2_m, sigma2, X2, preGamma[0,2,3,1], preExp1, 1 ).T )
            else:
                matGrad = (np.prod(S) * np.sqrt(np.pi) / (8 * np.prod(m) * np.prod(omega))) \
                        * (preKernel
                           + sigma
                             * (lfmGradientSigmaH3(gamma1_p, gamma1_m, sigma2, X, X2, preFactors[0,1], 1)\
                             + lfmGradientSigmaH3(gamma2_p, gamma2_m, sigma2, X2, X, preFactors[2,3], 1).T\
                             + lfmGradientSigmaH4(gamma1_p, gamma1_m, sigma2, X, preGamma[0,1,3,2], preExp2, 1 )\
                             + lfmGradientSigmaH4(gamma2_p, gamma2_m, sigma2, X2, preGamma[0,2,3,1], preExp1, 1 ).T ))


        if subComponent:
            if np.shape(meanVector)[0] == 1:
                matGrad = matGrad * meanVector
            else:
                matGrad = (meanVector * matGrad).T

        g1[3] = sum(sum(matGrad * covGrad)) * (-np.power(sigma, 3) / 4)
        g2[3] = g1[3]

        # Gradients with respect to S

        if np.all(np.isreal(omega)):
            if self.isNormalised[0]:
                matGrad = (1 / (4 * np.sqrt(2) * np.prod(m) * np.prod(omega))) * np.real(preKernel)
            else:
                matGrad = (sigma * np.sqrt(np.pi) / (4 * np.prod(m) * np.prod(omega))) * np.real(preKernel)
        else:
            if self.isNormalised[1]:
                matGrad = (1 / (8 * np.sqrt(2) * np.prod(m) * np.prod(omega))) * (preKernel)
            else:
                matGrad = (sigma * np.sqrt(np.pi) / (8 * np.prod(m) * np.prod(omega))) * (preKernel)


        if subComponent:
            if np.shape(meanVector)[0] == 1:
                matGrad = matGrad * meanVector
            else:
                    matGrad = (meanVector * matGrad).T
        # TODO
        g1[4:] = sum(sum(S[1] * matGrad * covGrad))
        g2[4:] = sum(sum(S[0] * matGrad * covGrad))

        g2[3] = 0  # Otherwise is counted twice

        g1 = np.real(g1)
        g2 = np.real(g2)
        # names = {'mass', 'spring', 'damper', 'inverse width', 'sensitivity'}
        # scale = 2/inverse width

        return [g1, g2]
    

    def reset_gradients(self):
        self.scale.gradient = np.zeros_like(self.scale.gradient)
        self.mass.gradient = np.zeros_like(self.mass.gradient)
        self.spring.gradient = np.zeros_like(self.spring.gradient)
        self.damper.gradient = np.zeros_like(self.damper.gradient)
        self.sensitivity.gradient = np.zeros_like(self.sensitivity.gradient)