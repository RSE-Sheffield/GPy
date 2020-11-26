from GPy.kern import Kern
from GPy.kern.src.lfm import *
import numpy as np
from GPy.core.parameterization import Param
from paramz.transformations import Logexp
from GPy.kern.src.independent_outputs import index_to_slices

from GPy.util.config import config # for assesing whether to use cython

class LFMXRBF(Kern):
    """
    LFM X RBF convolved kernel todo: expand.
    """

    def __init__(self, input_dim, active_dims=None, scale=None, mass=None, spring=None, damper=None, sensitivity=None, isNormalised=None, name='lfmXrbf'):

        super(LFMXRBF, self).__init__(input_dim, active_dims, name)

        if scale is None:
            scale =  2
        self.scale = Param('scale', scale)

        if mass is None:
            mass =  1
        self.mass = Param('mass', mass)

        if spring is None:
            spring =  1
        self.spring = Param('spring', spring)

        if damper is None:
            damper = 1
        self.damper = Param('damper', damper)

        if sensitivity is None:
            sensitivity = 1
        self.sensitivity = Param('sensitivity', sensitivity)

        self.link_parameters(self.scale, self.mass, self.spring, self.damper, self.sensitivity)

        if isNormalised is None:
            isNormalised = [False for _ in range(2)]
        self.isNormalised = isNormalised

        self.recalculate_intermediate_variables()

    def recalculate_intermediate_variables(self):
        # alpha and omega are intermediate variables used in the model and gradient for optimisation
        self.alpha = self.damper / (2 * self.mass)
        self.omega = np.sqrt(self.spring / self.mass - self.alpha * self.alpha)
        self.omega_isreal = np.isreal(self.omega).all()

        self.gamma = self.alpha + 1j * self.omega

    def K(self, X1, X2=None):
            # LFMXRBFKERNCOMPUTE Compute a cross kernel between the LFM and RBF kernels.
            # FORMAT
            # DESC computes cross kernel terms between LFM and RBF kernels for
            # the multiple output kernel.
            # ARG lfmKern : the kernel structure associated with the LFM
            # kernel.
            # ARG rbfKern : the kernel structure associated with the RBF
            # kernel.
            # ARG t : inputs for which kernel is to be computed.
            # RETURN K : block of values from kernel matrix.
            #
            # FORMAT
            # DESC computes cross kernel terms between LFM and RBF kernels for
            # the multiple output kernel.
            # ARG lfmKern : the kernel structure associated with the LFM
            # kernel.
            # ARG rbfKern : the kernel structure associated with the RBF
            # kernel.
            # ARG t1 : row inputs for which kernel is to be computed.
            # ARG t2 : column inputs for which kernel is to be computed.
            # RETURN K : block of values from kernel matrix.
            #
            # SEEALSO : multiKernParamInit, multiKernCompute, lfmKernParamInit, rbfKernParamInit
            #
            # COPYRIGHT : David Luengo, 2007, 2008, Mauricio Alvarez, 2008
            #
            # MODIFICATIONS : Neil D. Lawrence, 2007

            # KERN

            if X2 is None:
                X2 = X1
            assert X1.shape[1] == 1 and X2.shape[1] == 1, 'Input of' + inspect.stack()[0][3]  + 'can only have one column'

            # Kernel evaluation
            if self.omega_isreal:
                gamma = self.alpha + 1j * self.omega
                sK = np.imag(lfmUpsilonMatrix(gamma, self.scale, X1, X2))
                K0 = np.sqrt(np.pi) * np.sqrt(self.scale) * self.sensitivity / (2 * self.mass * self.omega)
                K = -K0 * sK

            else:
                gamma1 = self.alpha + 1j * self.omega
                gamma2 = self.alpha - 1j * self.omega
                sK = lfmUpsilonMatrix(gamma2, self.scale, X1, X2) - lfmUpsilonMatrix(gamma1, self.scale, X1, X2)
                K0 = np.sqrt(np.pi) * np.sqrt(self.scale) * self.sensitivity / (1j * 4 * self.mass * self.omega)
                K = K0 * sK
            return K

    def update_gradients_full(self, X1, X2=None, dL_dK=None, meanVector=None):

        # LFMXRBFKERNGRADIENT Compute gradient between the LFM and RBF kernels.
        # FORMAT
        # DESC computes the gradient of an objective function with respect
        # to cross kernel terms between LFM and RBF kernels for
        # the multiple output kernel.
        # ARG lfmKern : the kernel structure associated with the LFM
        # kernel.
        # ARG rbfKern : the kernel structure associated with the RBF
        # kernel.
        # ARG t : inputs for which kernel is to be computed.
        # RETURN g1 : gradient of objective function with respect to kernel
        # parameters of LFM kernel.
        # RETURN g2 : gradient of objective function with respect to kernel
        # parameters of RBF kernel.
        #
        # FORMAT
        # DESC computes the gradient of an objective function with respect
        # to cross kernel terms between LFM and RBF kernels for
        # the multiple output kernel.
        # ARG lfmKern : the kernel structure associated with the LFM
        # kernel.
        # ARG rbfKern : the kernel structure associated with the RBF
        # kernel.
        # ARG X1 : row inputs for which kernel is to be computed.
        # ARG t2 : column inputs for which kernel is to be computed.
        # RETURN g1 : gradient of objective function with respect to kernel
        # parameters of LFM kernel.
        # RETURN g2 : gradient of objective function with respect to kernel
        # parameters of RBF kernel.
        #
        # FORMAT
        # DESC computes the gradient of an objective function with respect
        # to cross kernel terms between LFM and RBF kernels for
        # the multiple output kernel.
        # ARG lfmKern : the kernel structure associated with the LFM
        # kernel.
        # ARG rbfKern : the kernel structure associated with the RBF
        # kernel.
        # ARG X1 : row inputs for which kernel is to be computed.
        # ARG t2 : column inputs for which kernel is to be computed.
        # ARG meanVec : precomputed factor that is used for the switching dynamical
        # latent force model.
        # RETURN g1 : gradient of objective function with respect to kernel
        # parameters of LFM kernel.
        # RETURN g2 : gradient of objective function with respect to kernel
        # parameters of RBF kernel.
        #
        # SEEALSO : multiKernParamInit, multiKernCompute, lfmKernParamInit, rbfKernParamInit
        #
        # COPYRIGHT : David Luengo, 2007, 2008
        #
        # MODIFICATIONS : Neil D. Lawrence, 2007
        #
        # MODIFICATIONS : Mauricio A. Alvarez, 2008, 2010

        # KERN

        if dL_dK is None:
            # this section of codes are historical legacy from the Matlab codes. I don't think it actually functioning
            # here. --Tianqi
            dL_dK  = X2
            X2 = X1
        if meanVector is not None:
            if np.prod(meanVector.shape) > 1:
                if meanVector.shape[0] == 1:
                    assert meanVector.shape[1] == dL_dK.shape[1], 'The dimensions of meanVector don''t correspond to the dimensions of dL_dK.'
                else:
                    assert meanVector.shape[1] == dL_dK.shape[1], 'The dimensions of meanVector don''t correspond to the dimensions of dL_dK'
            else:
                if np.prod(X1.shape) == 1 and np.prod(X2.shape)  > 1:
                    # matGrad will be row vector and so should be dL_dK
                    dimdL_dK = len(dL_dK)
                    dL_dK = dL_dK.reshape([1,dimdL_dK])
                elif np.prod(X1.shape) > 1 and np.prod(X2.shape) == 1:
                    # matGrad will be column vector and sp should be dL_dK
                    dimdL_dK = len(dL_dK)
                    dL_dK = dL_dK.reshape([dimdL_dK, 1])

        assert X1.shape[1] == 1 and X2.shape[1] == 1, 'Input can only have one column'
    
        m = self.mass
        D = self.spring
        C = self.damper
        S = self.sensitivity
        
        if self.omega_isreal:
            ComputeUpsilon1 = lfmUpsilonMatrix(self.gamma, sigma2, X1, X2)
            if self.unilateral_kernels[q1].isNormalised:
                K0 = self.sensitivity / (2 * np.sqrt(2) * self.mass * self.omega)
            else:
                K0 = np.sqrt(self.scale) * np.sqrt(np.pi) * self.sensitivity / (2 * self.mass * self.omega)
        else:
            gamma1 = self.alpha + 1j * self.omega
            gamma2 = self.alpha - 1j * self.omega
            ComputeUpsilon1 = lfmUpsilonMatrix(gamma2, self.scale, X1, X2)
            ComputeUpsilon2 = lfmUpsilonMatrix(gamma1, self.scale, X1, X2)
            if self.isNormalised:
                K0 = (S / (1j * 4 * csqrt(2) * m * self.omega))
            else:
                K0 = np.sqrt(self.scale) * np.sqrt(np.pi) * S / (1j * 4 * m * self.omega)

        g1 = np.zeros((5))
        g2 = np.zeros((2))
        # Gradient with respect to m, D and C
        for ind in np.arange(3):  # Parameter (m, D or C)
            # Choosing the right gradients for m, omega, gamma1 and gamma2
            if ind == 0:  # Gradient wrt m
                gradThetaM = 1
                gradThetaAlpha = -C / (2 * (m ** 2))
                gradThetaOmega = (C ** 2 - 2 * m * D) / (2 * (m ** 2) * np.sqrt(4 * m * D - C ** 2))
            if ind == 1:  # Gradient wrt D
                gradThetaM = 0
                gradThetaAlpha = 0
                gradThetaOmega = 1 / np.sqrt(4 * m * D - C ** 2)
            if ind == 2:  # Gradient wrt C
                gradThetaM = 0
                gradThetaAlpha = 1 / (2 * m)
                gradThetaOmega = -C / (2 * m * np.sqrt(4 * m * D - C ** 2))

            # Gradient evaluation

            if self.omega_isreal:
                gradThetaGamma = gradThetaAlpha + 1j * gradThetaOmega
                matGrad = -K0 * np.imag(lfmGradientUpsilonMatrix(self.gamma, self.scale, X1, X2) * gradThetaGamma \
                                     - (gradThetaM / m + gradThetaOmega / self.omega) \
                                     * ComputeUpsilon1)
            else:
                gamma1 = self.alpha + 1j * self.omega
                gamma2 = self.alpha - 1j * self.omega
                gradThetaGamma1 = gradThetaAlpha + 1j * gradThetaOmega
                gradThetaGamma2 = gradThetaAlpha - 1j * gradThetaOmega
                matGrad = K0 * (lfmGradientUpsilonMatrix(gamma2, self.scale, X1, X2) * gradThetaGamma2 \
                                - lfmGradientUpsilonMatrix(gamma1, self.scale, X1, X2) * gradThetaGamma1 \
                                - (gradThetaM / m + gradThetaOmega / self.omega) \
                                * (ComputeUpsilon1 - ComputeUpsilon2))

            g1[ind] = sum(sum(matGrad * dL_dK))

        # Gradient with respect to sigma
    
        if self.omega_isreal:
            if self.isNormalised:
               matGrad = -K0 * np.imag(lfmGradientSigmaUpsilonMatrix(gamma, self.scale, X1, X2))
            else:
                matGrad = -(np.sqrt(np.pi) * S / (2 * m * self.omega)) \
                * np.imag(ComputeUpsilon1 \
                          + np.sqrt(self.scale) * lfmGradientSigmaUpsilonMatrix(gamma, self.scale, X1, X2))
        else:
            gamma1 = self.alpha + 1j * self.omega
            gamma2 = self.alpha - 1j * self.omega
            if self.isNormalised:
                matGrad = K0 * (lfmGradientSigmaUpsilonMatrix(gamma2, self.scale, X1, X2) \
                - lfmGradientSigmaUpsilonMatrix(gamma1, self.scale, X1, X2))
            else:
                matGrad = (np.sqrt(np.pi) * S / (1j * 4 * m * self.omega)) \
                * (ComputeUpsilon1 - ComputeUpsilon2 \
                + np.sqrt(self.scale) * (lfmGradientSigmaUpsilonMatrix(gamma2, self.scale, X1, X2) \
                           - lfmGradientSigmaUpsilonMatrix(gamma1, self.scale, X1, X2)))

        g1[3] = sum(sum(matGrad * dL_dK)) * (-(sigma ** 3) / 4)  # temporarly introduced by MA
        g2[0] = g1[3]
    
        # Gradient with respect to S
    
        if self.omega_isreal:
            if self.isNormalised:
                matGrad = -(1 / (2 * np.sqrt(2) * m * self.omega)) * np.imag(ComputeUpsilon1)
            else:
                matGrad = -(np.sqrt(np.pi) * np.sqrt(self.scale) / (2 * m * self.omega)) * np.imag(ComputeUpsilon1)
        else:
            if self.isNormalised:
                matGrad = (1 / (1j * 4 * np.sqrt(2) * m * self.omega)) * (ComputeUpsilon1 - ComputeUpsilon2)
            else:
                matGrad = (np.sqrt(np.pi) * np.sqrt(self.scale) / (1j * 4 * m * self.omega)) * (ComputeUpsilon1 - ComputeUpsilon2)
    
        g1[4] = sum(sum(matGrad * dL_dK))
        g1 = np.real(g1)
        # Gradient with respect to the "variance" of the RBF
        g2[0] = 0  # Otherwise is counted twice, temporarly changed by MA
        g2[1] = 0
        return [g1, g2]