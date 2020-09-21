# This is currently completely broken, but aspires to be a representation of the
# lfmXrbf convolved kernel for use in latent force models.

class LFMXRBF(Kern):

    def __init__(self, input_dim, output_dim, name='lfmXrbf'):
        pass

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
            assert X1.shape[1] == 1 and X2.shape[1] == 1, 'Input of' + inspect.stack()[0][3]  +'can only have one column' # _update_gradients_LFMXRBF.__name__
            assert self.unilateral_kernels[q1].inv_l == self.unilateral_kernels[q1].inv_l, \
            'Kernels cannot be cross combined if they have different inverse widths.'

            # Get length scale out.
            sigma2 = self.unilateral_kernels[q1].sigma2
            sigma = self.unilateral_kernels[q1].sigma

            # Parameters of the kernel
            alpha = self.unilateral_kernels[q1].alpha
            omega = self.unilateral_kernels[q1].omega

            # Kernel evaluation
            if self.unilateral_kernels[q1].omega_isreal:
                gamma = alpha + 1j * omega
                sK = np.imag(lfmComputeUpsilonMatrix(gamma, sigma2, X1, X2))
                K0 = csqrt(np.pi) * sigma * self.unilateral_kernels[q1].sensitivity \
                    / (2 * self.unilateral_kernels[q1].mass * omega)
                K = -K0 * sK

            else:
                gamma1 = alpha + 1j * omega
                gamma2 = alpha - 1j * omega
                sK = lfmComputeUpsilonMatrix(gamma2, sigma2, X1, X2) - lfmComputeUpsilonMatrix(gamma1, sigma2, X1, X2)
                K0 = csqrt(np.pi) * sigma * self.unilateral_kernels[q1].sensitivity \
                    / (1j * 4 * self.unilateral_kernels[q1].mass * omega)
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

        subComponent = False  # This is just a flag that indicates if this kernel is part of a bigger kernel (SDLFM)

        if dL_dK is None:
            # this section of codes are historical legacy from the Matlab codes. I don't think it actually functioning
            # here. --Tianqi
            dL_dK  = X2
            X2 = X1
        if meanVector is not None:
            subComponent = True
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
        assert self.unilateral_kernels[q1].inv_l == self.unilateral_kernels[q2].inv_l, \
                'Kernels cannot be cross combined if they have different inverse widths.'
    
        m = self.unilateral_kernels[q1].mass
        D = self.unilateral_kernels[q1].spring
        C = self.unilateral_kernels[q1].damper
        S = self.unilateral_kernels[q1].sensitivity
    
        alpha = C / (2 * m)
        omega = csqrt(D / m-alpha ** 2)
    
        sigma2 = 2 / self.unilateral_kernels[q1].inv_l
        sigma = csqrt(sigma2)
    
        if np.isreal(omega):
            gamma = alpha + 1j * omega
            ComputeUpsilon1 = lfmComputeUpsilonMatrix(gamma, sigma2, X1, X2)
            if self.unilateral_kernels[q1].isNormalised:
                K0 = S / (2 * csqrt(2) * m * omega)
            else:
                K0 = sigma * csqrt(np.pi) * S / (2 * m * omega)
        else:
            gamma1 = alpha + 1j * omega
            gamma2 = alpha - 1j * omega
            ComputeUpsilon1 = lfmComputeUpsilonMatrix(gamma2, sigma2, X1, X2)
            ComputeUpsilon2 = lfmComputeUpsilonMatrix(gamma1, sigma2, X1, X2)
            if self.unilateral_kernels[q1].isNormalised:
                K0 = (S / (1j * 4 * csqrt(2) * m * omega))
            else:
                K0 = sigma * csqrt(np.pi) * S / (1j * 4 * m * omega)

        g1 = np.zeros((5))
        g2 = np.zeros((2))
        # Gradient with respect to m, D and C
        for ind in np.arange(3):  # Parameter (m, D or C)
            # Choosing the right gradients for m, omega, gamma1 and gamma2
            if ind == 0:  # Gradient wrt m
                gradThetaM = 1
                gradThetaAlpha = -C / (2 * (m ** 2))
                gradThetaOmega = (C ** 2 - 2 * m * D) / (2 * (m ** 2) * csqrt(4 * m * D - C ** 2))
            if ind == 1:  # Gradient wrt D
                gradThetaM = 0
                gradThetaAlpha = 0
                gradThetaOmega = 1 / csqrt(4 * m * D - C ** 2)
            if ind == 2:  # Gradient wrt C
                gradThetaM = 0
                gradThetaAlpha = 1 / (2 * m)
                gradThetaOmega = -C / (2 * m * csqrt(4 * m * D - C ** 2))

            # Gradient evaluation

            if np.isreal(omega):
                gamma = alpha + 1j * omega
                gradThetaGamma = gradThetaAlpha + 1j * gradThetaOmega
                matGrad = -K0 * np.imag(lfmGradientUpsilonMatrix(gamma, sigma2, X1, X2) * gradThetaGamma \
                                     - (gradThetaM / m + gradThetaOmega / omega) \
                                     * ComputeUpsilon1)
            else:
                gamma1 = alpha + 1j * omega
                gamma2 = alpha - 1j * omega
                gradThetaGamma1 = gradThetaAlpha + 1j * gradThetaOmega
                gradThetaGamma2 = gradThetaAlpha - 1j * gradThetaOmega
                matGrad = K0 * (lfmGradientUpsilonMatrix(gamma2, sigma2, X1, X2) * gradThetaGamma2 \
                                - lfmGradientUpsilonMatrix(gamma1, sigma2, X1, X2) * gradThetaGamma1 \
                                - (gradThetaM / self.unilateral_kernels[q1].mass + gradThetaOmega / omega) \
                                * (ComputeUpsilon1 - ComputeUpsilon2))

            if subComponent:
                if meanVector.shape[1] == 1:
                    matGrad = matGrad * meanVector
                else:
                    matGrad = (meanVector * matGrad).T
            g1[ind] = sum(sum(matGrad * dL_dK))

    
        # Gradient with respect to sigma
    
        if np.isreal(omega):
            gamma = alpha + 1j * omega
            if self.unilateral_kernels[q1].isNormalised:
               matGrad = -K0 * np.imag(lfmGradientSigmaUpsilonMatrix(gamma, sigma2, X1, X2))
            else:
                matGrad = -(csqrt(np.pi) * S / (2 * m * omega)) \
                * np.imag(ComputeUpsilon1 \
                          + sigma * lfmGradientSigmaUpsilonMatrix(gamma, sigma2, X1, X2))
        else:
            gamma1 = alpha + 1j * omega
            gamma2 = alpha - 1j * omega
            if self.unilateral_kernels[q1].isNormalised:
                matGrad = K0 * (lfmGradientSigmaUpsilonMatrix(gamma2, sigma2, X1, X2) \
                - lfmGradientSigmaUpsilonMatrix(gamma1, sigma2, X1, X2))
            else:
                matGrad = (csqrt(np.pi) * S / (1j * 4 * m * omega)) \
                * (ComputeUpsilon1 - ComputeUpsilon2 \
                + sigma * (lfmGradientSigmaUpsilonMatrix(gamma2, sigma2, X1, X2) \
                           - lfmGradientSigmaUpsilonMatrix(gamma1, sigma2, X1, X2)))

    
        if subComponent:
            if meanVector.shape[0] == 1:
              matGrad = matGrad * meanVector
            else:
              matGrad = (meanVector * matGrad).T

        g1[3] = sum(sum(matGrad * dL_dK)) * (-(sigma ** 3) / 4)  # temporarly introduced by MA
        g2[0] = g1[3]
    
        # Gradient with respect to S
    
        if np.isreal(omega):
            if self.unilateral_kernels[q1].isNormalised:
                matGrad = -(1 / (2 * csqrt(2) * m * omega)) * np.imag(ComputeUpsilon1)
            else:
                matGrad = -(csqrt(np.pi) * sigma / (2 * m * omega)) * np.imag(ComputeUpsilon1)
        else:
            if self.unilateral_kernels[q1].isNormalised:
                matGrad = (1 / (1j * 4 * csqrt(2) * m * omega)) * (ComputeUpsilon1 - ComputeUpsilon2)
            else:
                matGrad = (csqrt(np.pi) * sigma / (1j * 4 * m * omega)) * (ComputeUpsilon1 - ComputeUpsilon2)

    
        if subComponent:
            if meanVector.shape[0] == 1:
                matGrad = matGrad * meanVector
            else:
                matGrad = (meanVector * matGrad).T

    
        g1[4] = sum(sum(matGrad * dL_dK))
        g1 = np.real(g1)
        # Gradient with respect to the "variance" of the RBF
        g2[0] = 0  # Otherwise is counted twice, temporarly changed by MA
        g2[1] = 0
        return [g1, g2]