# Copyright (c) 2012, James Hensman and Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)
# import sys

"""
This module adds functionality for Latent Force Models (LFM) to GPy. **It is partially complete.**

LFMs are a form of multiple output GP that make use of a kernel function inspired by a differential equation representing a phycial process (see M. Alvarez, D. Luengo and N. D. Lawrence, "Latent Force Models", Proc. AISTATS 2009.).

LFMs combine kernels using convolution and will require functionality in :py:class:`GPy.kern.MultioutputKern` to be combined into a useful model..

Multiple output GPs using co-regionalised regression are implemented elsewhere in GPy (:py:class:`GPy.models.GPCoregionalizedRegression`). 
"""

import numpy as np
from . import lfm_C

def cell(d0, d1):
    """The purpose of this function is unknown."""
    if d1 == 1:
        return [None for _ in range(d0)]
    else:
        return [[None for _ in range(d1)] for _ in range(d0)]

def lfmUpsilonMatrix(gamma1_p, sigma2, X, X2):
    """Computes the Upsilon's Gradient wrt to gamma.
        
        :param gamma: gamma value system
        :param sigma2: squared lengthscale
        :param X: first time input
        :param X2: second time input

        :return: gradient Matrix
    """
    return lfm_C.UpsilonMatrix(gamma1_p, sigma2, X.astype(np.float64), X2.astype(np.float64))

def lfmUpsilonVector(gamma1_p, sigma2, X):
    """Computes Upsilon given a input vector
    
    :param gamma: gamma value system
    :param sigma2: squared lengthscale
    :param X: first time input

    :return: upsilon vector
    """
    return lfm_C.UpsilonVector(gamma1_p, sigma2, X.astype(np.float64))

def lfmGradientUpsilonMatrix(gamma1_p, sigma2, X, X2):
    """Computes the Upsilon's Gradient wrt to gamma.

    :param gamma: gamma value system
    :param sigma2: squared lengthscale
    :param X: first time input
    :param X2: second time input

    :return: gradient Matrix
    """
    return lfm_C.GradientUpsilonMatrix(gamma1_p, sigma2, X.astype(np.float64), X2.astype(np.float64))

def lfmGradientUpsilonVector(gamma1_p, sigma2, X):
    """Computes the Upsilon's Gradient wrt to Sigma assuming that X2 is zero vector.

    :param gamma: gamma value system
    :param sigma2: squared lengthscale
    :param t1: first time input
    
    :return: gradient vector
    """
    return lfm_C.GradientUpsilonVector(gamma1_p, sigma2, X.astype(np.float64))

def lfmGradientSigmaUpsilonMatrix(gamma1_p, sigma2, X, X2):
    """
    Computes the Upsilon's Gradient wrt to Sigma.
    
    :param gamma: gamma value system
    :param sigma2: squared lengthscale
    :param X: first time input
    :param X2: second time input
    
    :return gradient Matrix    
    """
    return lfm_C.GradientSigmaUpsilonMatrix(gamma1_p, sigma2, X.astype(np.float64), X2.astype(np.float64))

def lfmGradientSigmaUpsilonVector(gamma1_p, sigma2, X):
    """
    Computes the Upsilon's Gradient wrt to Sigma assuming that t2 is zero vector.

    :param gamma: gamma value system
    :param  sigma2: squared lengthscale
    :param  X: first time input
    
    :return gradient vector (x 1)
    """
    return lfm_C.GradientSigmaUpsilonVector(gamma1_p, sigma2, X.astype(np.float64))

def lfmComputeH3( gamma1_p, gamma1_m, sigma2, X, X2, preFactor, mode=None, term=None):
    """
    Helper function for computing part of the LFM kernel.
    
    Computes a portion of the LFM kernel.

    :param gamma1: Gamma value for first system.
    :param gamma2: Gamma value for second system.
    :param sigma2: length scale of latent process.
    :param X: first time input.
    :param X2: second time input.
    :param mode: indicates in which way the vectors X and X2 must be transposed
    :param term: the purpose of this parameter is unknown.

    :return h: result of this subcomponent of the kernel for the given values.
    """
    if not mode:
        if not term:
            upsilon = lfmUpsilonMatrix(gamma1_p, sigma2, X, X2)
            h = preFactor * upsilon
        else:
            upsilon = lfmUpsilonMatrix(gamma1_p, sigma2, X, X2)
            h = -preFactor[0] * upsilon + preFactor[1] * np.conj(upsilon)

    else:

        upsilon = np.hstack([lfmUpsilonMatrix(gamma1_p, sigma2, X, X2), lfmUpsilonMatrix(gamma1_m, sigma2, X, X2)])
        h = preFactor[0] * upsilon + preFactor[1] * upsilon
    return [h, upsilon]

def lfmComputeH4(gamma1_p, gamma1_m, sigma2, X, preFactor, preExp, mode=None, term=None ):
    """
    Helper function for computing part of the LFM kernel.
    
    Computes a portion of the LFM kernel.

    :param gamma1: Gamma value for first system.
    :param gamma2: Gamma value for second system.
    :param sigma2: length scale of latent process.
    :param X: first time input.
    :param preFactor: precomputed constants.
    :param preExp: precomputed exponentials.
    :param mode: indicates in which way the vectors X and X2 must be transposed.
    :param term: the purpose of this parameter is unknown.

    :return h: result of this subcomponent of the kernel for the given values.
    """
    if not mode:
        if not term:
            upsilon = lfmUpsilonVector(gamma1_p, sigma2, X)[:, None]
            h = np.matmul(upsilon, (preExp / preFactor[0] - np.conj(preExp) / preFactor[1]).T)

        else:
            upsilon = lfmUpsilonVector(gamma1_p, sigma2, X)[:, None]
            h = np.matmul(upsilon, (preExp / preFactor[0]).T) - np.matmul(np.conj(upsilon), (preExp/preFactor[1]).T)

    else:
        upsilon = [lfmUpsilonVector(gamma1_p, sigma2, X)[:, None], lfmUpsilonVector(gamma1_m, sigma2, X)[:, None]]
        h = np.matmul(upsilon[0], (preExp[:, 0] / preFactor[0] - preExp[:, 1] / preFactor[1]).T) \
            + np.matmul(upsilon[1] * (preExp[:, 1] / preFactor[2] - preExp[:, 0] / preFactor[3]).T)
    return [h, upsilon]

def lfmGradientH31(preFactor, preFactorGrad, gradThetaGamma, gradUpsilon1, gradUpsilon2, compUpsilon1, compUpsilon2, mode, term=None):
    """
    Gradient of the function :math:`h_i(z)` with respect to some of the
    hyperparameters of the kernel: :math:`m_k`, :math:`C_k`, :math:`D_k`, :math:`m_r`, :math:`C_r` or :math:`D_r`.
    
    Computes the gradient of the function :math:`h_i(z)` with respect to some of
    the parameters of the system (mass, spring or damper).

    :param gamma1: Gamma value for first system.
    :param gamma2: Gamma value for second system.
    :param sigma2: length scale of latent process.
    :param gradThetaGamma: Vector with the gradient of gamma1 and gamma2 with respect to the desired parameter.
    :param X: first time input.
    :param X2: second time input.
    :param mode: indicates in which way the vectors X and X2 must be transposed.

    :return g: Gradient of the function with respect to the desired parameter.
    """

    #this has come from matlab where everything is an array, so let's make everything an array for now
    preFactor = np.atleast_1d(preFactor)
    preFactorGrad = np.atleast_1d(preFactorGrad)

    if not mode:
        if not term:
            g = (preFactor[0]*gradUpsilon1 + preFactorGrad[0]*compUpsilon1)*gradThetaGamma
        else:
            g = (-preFactor[0]*gradUpsilon1 + preFactorGrad[0]*compUpsilon1)*gradThetaGamma[0] \
                +(preFactor[1]*np.conj(gradUpsilon1) - preFactorGrad[1]*np.conj(compUpsilon1))*gradThetaGamma[1]
    else:
        g = (preFactor[0]*gradUpsilon1 + preFactorGrad[0]*compUpsilon1)*gradThetaGamma[0] \
            + (preFactor[1]*gradUpsilon2 + preFactorGrad[1]*compUpsilon2)*gradThetaGamma[1]
    return g

def lfmGradientH32(preFactor, gradThetaGamma, compUpsilon1,compUpsilon2, mode, term=None):
    """
    Gradient of the function :math:`h_i(z)` with respect to some of the
    hyperparameters of the kernel: :math:`m_k`, :math:`C_k`, :math:`D_k`, :math:`m_r`, :math:`C_r` or :math:`D_r`.

    Computes the gradient of the function :math:`h_i(z)` with respect to some of
    the parameters of the system (mass, spring or damper).

    :param gamma1: Gamma value for first system.
    :param gamma2: Gamma value for second system.
    :param sigma2: length scale of latent process.
    :param gradThetaGamma: Vector with the gradient of gamma1 and gamma2 with respect to the desired parameter.
    :param X: first time input.
    :param X2: second time input.
    :param mode: indicates in which way the vectors X and X2 must be transposed

    :return g: Gradient of the function with respect to the desired parameter.
    """

    if not mode:
        if not term:
            g = compUpsilon1*(-(gradThetaGamma[1]/preFactor[1]) + (gradThetaGamma[0]/preFactor[0]))
        else:
            g = (compUpsilon1*preFactor[0] - np.conj(compUpsilon1)*preFactor[1])*gradThetaGamma
    else:
        g = compUpsilon1*(-(gradThetaGamma[1]/preFactor[2]) + (gradThetaGamma[0]/preFactor[0])) \
            + compUpsilon2*(-(gradThetaGamma[0]/preFactor[1]) + (gradThetaGamma[1]/preFactor[3]))
    return g

def lfmGradientH41(preFactor, preFactorGrad, gradThetaGamma, preExp, gradUpsilon1, gradUpsilon2, compUpsilon1, compUpsilon2, mode, term=None):
    """
    Gradient of the function :math:`h_i(z)` with respect to some of the
    hyperparameters of the kernel: :math:`m_k`, :math:`C_k`, :math:`D_k`, :math:`m_r`, :math:`C_r` or :math:`D_r`.

    Computes the gradient of the function :math:`h_i(z)` with respect to some of
    the parameters of the system (mass, spring or damper).
    
    :param gamma1: Gamma value for first system.
    :param gamma2: Gamma value for second system.
    :param sigma2: length scale of latent process.
    :param gradThetaGamma: Vector with the gradient of gamma1 and gamma2 with respect to the desired parameter.
    :param X: first time input.
    :param X2: second time input.
    :param mode: indicates in which way the vectors X and X2 must be transposed.

    :return g: Gradient of the function with respect to the desired parameter.
    """

    if not mode:
        if not term:
            g = (((gradUpsilon1 * gradThetaGamma) * (preExp/preFactor[0] - np.conj(preExp)/preFactor[1])).T \
                + np.outer((compUpsilon1 * gradThetaGamma), (- preExp/preFactorGrad[0] + np.conj(preExp)/preFactorGrad[1]).T)).T
        else:
            g = (((gradUpsilon1 * gradThetaGamma[0]) * (preExp/preFactor[0])).T \
                + np.outer((compUpsilon1 * gradThetaGamma[0]), (- preExp/preFactorGrad[0]).T) \
                + ((np.conj(gradUpsilon1) * gradThetaGamma[1]) * (- preExp/preFactor[1])).T \
                + np.outer((np.conj(compUpsilon1) * gradThetaGamma[1]), (preExp/preFactorGrad[1]).T)).T
    else:
        g = (gradUpsilon1 * gradThetaGamma[0]) * (preExp[:, 0]/preFactor[0] - preExp[:, 1]/preFactor[1]) \
            + (compUpsilon1 * gradThetaGamma[0]) * (- preExp[:, 0]/preFactorGrad[0] + preExp[:, 1]/preFactorGrad[1]) \
            + (gradUpsilon2 * gradThetaGamma[1]) * (preExp[:, 1]/preFactor[3] - preExp[:, 0]/preFactor[2]) \
            + (compUpsilon2 * gradThetaGamma[1]) * (- preExp[:, 1]/preFactorGrad[3] + preExp[:, 0]/preFactorGrad[2])
    return g.T

def lfmGradientH42(preFactor, preFactorGrad, gradThetaGamma, preExp, preExpt,
    compUpsilon1, compUpsilon2, mode, term=None):
    """
    Gradient of the function :math:`h_i(z)` with respect to some of the
    hyperparameters of the kernel: :math:`m_k`, :math:`C_k`, :math:`D_k`, :math:`m_r`, :math:`C_r` or :math:`D_r`.

    Computes the gradient of the function :math:`h_i(z)` with respect to some of
    the parameters of the system (mass, spring or damper).

    :param gamma1: Gamma value for first system.
    :param gamma2: Gamma value for second system.
    :param sigma2: length scale of latent process.
    :param gradThetaGamma: Vector with the gradient of gamma1 and gamma2 with respect to the desired parameter.
    :param X: first time input.
    :param X2: second time input.
    :param mode: indicates in which way the vectors X and X2 must be transposed
    
    :param g: Gradient of the function with respect to the desired parameter.
    """

    if not mode:
        if not term:
            g = (compUpsilon1*(- (preExp/preFactorGrad[0] + preExpt/preFactor[0])*gradThetaGamma[0]
                + (np.conj(preExp)/preFactorGrad[1] + np.conj(preExpt)/preFactor[1])*gradThetaGamma[1]).T).T
        else:
            g = (np.outer(compUpsilon1, (- (preExp/preFactorGrad[0] + preExpt/preFactor[0])*gradThetaGamma).T) \
                + np.outer(np.conj(compUpsilon1), ((preExp/preFactorGrad[1] + preExpt/preFactor[1])*gradThetaGamma).T)).T
    else:
        g = compUpsilon1*((preExp[:, 1]/preFactorGrad[2] + preExpt[:, 1]/preFactor[2])*gradThetaGamma[1]
            - (preExp[:, 0]/preFactorGrad[0] + preExpt[:, 0]/preFactor[0])*gradThetaGamma[0]) \
            - compUpsilon2*((preExp[:, 1]/preFactorGrad[3] + preExpt[:, 1]/preFactor[3])*gradThetaGamma[1]
            - (preExp[:, 0]/preFactorGrad[1] + preExpt[:, 0]/preFactor[1])*gradThetaGamma[0])
    return g.T

def lfmGradientSigmaH3(gamma1, gamma2, sigma2, X, X2, preFactor, mode, term=None):
    """
    Gradient of the function :math:`h_i(z)` with respect :math:`\sigma`.

    Computes the gradient of the function :math:`h_i(z)` with respect to the
    length-scale of the input "force", :math:`\sigma`.

    :param gamma1: Gamma value for first system.
    :param gamma2: Gamma value for second system.
    :param sigma2: length scale of latent process.
    :param X: first time input.
    :param X2: second time input.
    :param mode: indicates in which way the vectors X and X2 must be transposed

    :return g: Gradient of the function with respect to :math:`\sigma`.
    """

    if not mode:
        if not term:
            g = preFactor * lfmGradientSigmaUpsilonMatrix(gamma1, sigma2, X, X2)
        else:
            gradupsilon = lfmGradientSigmaUpsilonMatrix(gamma1, sigma2, X, X2)
            g = -preFactor[0] * gradupsilon + preFactor[1] * np.conj(gradupsilon)
    else:
        g = preFactor[0] * lfmGradientSigmaUpsilonMatrix(gamma1, sigma2, X, X2) + \
            preFactor[1] * lfmGradientSigmaUpsilonMatrix(gamma2, sigma2, X, X2)
    return g

def lfmGradientSigmaH4(gamma1, gamma2, sigma2, X, preFactor, preExp, mode, term):
    """
    Gradient of the function :math:`h_i(z)` with respect :math:`\sigma`.
    
    Computes the gradient of the function :math:`h_i(z)` with respect to the
    length-scale of the input "force", :math:`\sigma`.

    :param gamma1: Gamma value for first system.
    :param gamma2: Gamma value for second system.
    :param sigma2: length scale of latent process.
    :param X: first time input.
    :param X2: second time input.
    :param mode: indicates in which way the vectors X and X2 must be transposed.

    :return g: Gradient of the function with respect to :math:`\sigma`.
    """

    if not mode:
        if not term:
            g = np.outer(lfmGradientSigmaUpsilonVector(gamma1, sigma2, X),(preExp/preFactor[0] - np.conj(preExp)/preFactor[1]).T)
        else:
            gradupsilon = lfmGradientSigmaUpsilonVector(gamma1, sigma2, X)
            g = (gradupsilon * (preExp/preFactor[0])).T - (np.conj(gradupsilon)*(preExp/preFactor[1])).T
    else:
        g = lfmGradientSigmaUpsilonVector(gamma1, sigma2, X)*(preExp[:, 0]/preFactor[0] - preExp[:, 1]/preFactor[1]).T \
            + lfmGradientSigmaUpsilonVector(gamma2, sigma2, X)*(preExp[:, 1]/preFactor[2] - preExp[:, 0]/preFactor[3]).T
    return g