# This is currently completely broken, but aspires to be a representation of the
# rbfXrbf convolved kernel for use in latent force models.

class RBFXRBF(Kern):

    def __init__(self, input_dim, output_dim, name='rbfXrbf'):
        pass

    def K(self, X1, X2=None):
        # RBFKERNCOMPUTE Compute the RBF kernel given the parameters and X.
        # FORMAT
        # DESC computes the kernel parameters for the radial basis function
        # kernel given inputs associated with rows and columns.
        # ARG kern : the kernel structure for which the matrix is computed.
        # ARG x : the input matrix associated with the rows of the kernel.
        # ARG x2 : the input matrix associated with the columns of the kernel.
        # RETURN k : the kernel matrix computed at the given points.
        # RETURN sk : unscaled kernel matrix (i.e. only the exponential part).
        #
        # FORMAT
        # DESC computes the kernel matrix for the radial basis function
        # kernel given a design matrix of inputs.
        # ARG kern : the kernel structure for which the matrix is computed.
        # ARG x : input data matrix in the form of a design matrix.
        # RETURN k : the kernel matrix computed at the given points.
        # RETURN sk : unscaled kernel matrix (i.e. only the exponential part).
        #
        # SEEALSO : rbfKernParamInit, kernCompute, kernCreate, rbfKernDiagCompute
        #
        # COPYRIGHT : Neil D. Lawrence, 2004, 2005, 2006
        #
        # MODIFICATIONS : Mauricio Alvarez, 2009, David Luengo, 2009

        # KERN

        if X2 is None:
            n2 = 0
        else:
            n2 = np.linalg.norm(X1-X2)

        wi2 = .5 * self.unilateral_kernels[q1].inv_l
        sk = np. exp(-n2 * wi2)
        k = self.unilateral_kernels[q1].variation * sk
        if self.unilateral_kernels[q1].isNormalised is True:
            k = k * csqrt(self.unilateral_kernels[q1].inv_l / (2 * np.pi))
        return [k, sk, n2]

    def  update_gradients_full(self, X1, X2=None, dL_dK=None, meanVector=None):

        # RBFKERNGRADIENT Gradient of RBF kernel's parameters.
        # FORMAT
        # DESC computes the gradient of functions with respect to the
        # radial basis function
        # kernel's parameters. As well as the kernel structure and the
        # input positions, the user provides a matrix PARTIAL which gives
        # the partial derivatives of the function with respect to the
        # relevant elements of the kernel matrix.
        # ARG kern : the kernel structure for which the gradients are being
        # computed.
        # ARG x : the input locations for which the gradients are being
        # computed.
        # ARG partial : matrix of partial derivatives of the function of
        # interest with respect to the kernel matrix. The argument takes
        # the form of a square matrix of dimension  numData, where numData is
        # the number of rows in X.
        # RETURN g : gradients of the function of interest with respect to
        # the kernel parameters. The ordering of the vector should match
        # that provided by the function kernExtractParam.
        #
        # FORMAT
        # DESC computes the derivatives as above, but input locations are
        # now provided in two matrices associated with rows and columns of
        # the kernel matrix.
        # ARG kern : the kernel structure for which the gradients are being
        # computed.
        # ARG x1 : the input locations associated with the rows of the
        # kernel matrix.
        # ARG x2 : the input locations associated with the columns of the
        # kernel matrix.
        # ARG partial : matrix of partial derivatives of the function of
        # interest with respect to the kernel matrix. The matrix should
        # have the same number of rows as X1 and the same number of columns
        # as X2 has rows.
        # RETURN g : gradients of the function of interest with respect to
        # the kernel parameters.
        #
        # SEEALSO rbfKernParamInit, kernGradient, rbfKernDiagGradient, kernGradX
        #
        # COPYRIGHT : Neil D. Lawrence, 2004, 2005, 2006, 2009
        #
        # MODIFICATIONS : Mauricio Alvarez, 2009, David Luengo, 2009
    
        # KERN
    
        # The last argument is covGrad
        if X2 is None:
            [k, sk, dist2xx] = self.K_sub_matrix_RBFXRBF( q1, X1, q2)
        else:
            [k, sk, dist2xx] = self.K_sub_matrix_RBFXRBF( q1, X1, q2, X2)
        g = np.zeros((2))
        # if gK is cell then return cell of gs
        if self.unilateral_kernels[q1].isNormalised:
            g[0] = (0.5 * self.unilateral_kernels[q1].variation / csqrt(2 * np.pi)) * sum(sum(dL_dK * sk \
                   * (1 / csqrt(self.unilateral_kernels[q1].inv_l) - csqrt(self.unilateral_kernels[q1].inv_l) * dist2xx)))
            g[1] = csqrt(self.unilateral_kernels[q1].inv_l / (2 * np.pi)) * sum(sum(dL_dK * sk))
        else:
            g[0] = - 0.5 * sum(sum(dL_dK * k * dist2xx))
            g[1] = sum(sum(dL_dK * sk))

        assert not any(np.isnan(g)), 'g is NaN'
        return g