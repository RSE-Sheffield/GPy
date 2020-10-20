# Copyright (c) 2012, James Hensman and Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)
# import sys

import numpy as np
from . import lfm_C

def cell(d0, d1):
    if d1 == 1:
        return [None for _ in range(d0)]
    else:
        return [[None for _ in range(d1)] for _ in range(d0)]

def lfmUpsilonMatrix(gamma1_p, sigma2, X, X2):
    return lfm_C.UpsilonMatrix(gamma1_p, sigma2, X.astype(np.float64), X2.astype(np.float64))

def lfmUpsilonVector(gamma1_p, sigma2, X):
    return lfm_C.UpsilonVector(gamma1_p, sigma2, X.astype(np.float64))

def lfmGradientUpsilonMatrix(gamma1_p, sigma2, X, X2):
    return lfm_C.GradientUpsilonMatrix(gamma1_p, sigma2, X.astype(np.float64), X2.astype(np.float64))

def lfmGradientUpsilonVector(gamma1_p, sigma2, X):
    return lfm_C.GradientUpsilonVector(gamma1_p, sigma2, X.astype(np.float64))

def lfmGradientSigmaUpsilonMatrix(gamma1_p, sigma2, X, X2):
    return lfm_C.GradientSigmaUpsilonMatrix(gamma1_p, sigma2, X.astype(np.float64), X2.astype(np.float64))

def lfmGradientSigmaUpsilonVector(gamma1_p, sigma2, X):
    return lfm_C.GradientSigmaUpsilonVector(gamma1_p, sigma2, X.astype(np.float64))

def lfmComputeH3( gamma1_p, gamma1_m, sigma2, X, X2, preFactor, mode=None, term=None):
    # LFMCOMPUTEH3 Helper function for computing part of the LFM kernel.
    # FORMAT
    # DESC computes a portion of the LFM kernel.
    # ARG gamma1 : Gamma value for first system.
    # ARG gamma2 : Gamma value for second system.
    # ARG sigma2 : length scale of latent process.
    # ARG X : first time input (number of time points x 1).
    # ARG X2 : second time input (number of time points x 1).
    # ARG mode: indicates in which way the vectors X and X2 must be transposed
    # RETURN h : result of this subcomponent of the kernel for the given values.
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
    # LFMCOMPUTEH4 Helper function for computing part of the LFM kernel.
    # FORMAT
    # DESC computes a portion of the LFM kernel.
    # ARG gamma1 : Gamma value for first system.
    # ARG gamma2 : Gamma value for second system.
    # ARG sigma2 : length scale of latent process.
    # ARG X : first time input (number of time points x 1).
    # ARG preFactor : precomputed constants
    # ARG preExp : precomputed exponentials
    # ARG mode: indicates in which way the vectors X and X2 must be transposed
    # RETURN h : result of this subcomponent of the kernel for the given values.

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

    # LFMGRADIENTH31 Gradient of the function h_i(z) with respect to some of the
    # hyperparameters of the kernel: m_k, C_k, D_k, m_r, C_r or D_r.
    # FORMAT
    # DESC Computes the gradient of the function h_i(z) with respect to some of
    # the parameters of the system (mass, spring or damper).
    # ARG gamma1 : Gamma value for first system.
    # ARG gamma2 : Gamma value for second system.
    # ARG sigma2 : length scale of latent process.
    # ARG gradThetaGamma : Vector with the gradient of gamma1 and gamma2 with
    # respect to the desired parameter.
    # ARG X : first time input (number of time points x 1).
    # ARG X2 : second time input (number of time points x 1)
    # ARG mode: indicates in which way the vectors X and X2 must be transposed
    # RETURN g : Gradient of the function with respect to the desired
    # parameter.
    #
    # Author : Tianqi Wei
    # Based on Matlab codes by David Luengo, 2007, 2008, Mauricio Alvarez, 2008
    if not mode:
        if not term:
            g = (preFactor*gradUpsilon1 + preFactorGrad*compUpsilon1)*gradThetaGamma
        else:
            g = (-preFactor[0]*gradUpsilon1 + preFactorGrad[0]*compUpsilon1)*gradThetaGamma[0] \
                +(preFactor[1]*np.conj(gradUpsilon1) - preFactorGrad[1]*np.conj(compUpsilon1))*gradThetaGamma[1]
    else:
        g = (preFactor[0]*gradUpsilon1 + preFactorGrad[0]*compUpsilon1)*gradThetaGamma[0] \
            + (preFactor[1]*gradUpsilon2 + preFactorGrad[1]*compUpsilon2)*gradThetaGamma[1]
    return g

def lfmGradientH32(preFactor, gradThetaGamma, compUpsilon1,compUpsilon2, mode, term=None):
    # LFMGRADIENTH32 Gradient of the function h_i(z) with respect to some of the
    # hyperparameters of the kernel: m_k, C_k, D_k, m_r, C_r or D_r.
    # FORMAT
    # DESC Computes the gradient of the function h_i(z) with respect to some of
    # the parameters of the system (mass, spring or damper).
    # ARG gamma1 : Gamma value for first system.
    # ARG gamma2 : Gamma value for second system.
    # ARG sigma2 : length scale of latent process.
    # ARG gradThetaGamma : Vector with the gradient of gamma1 and gamma2 with
    # respect to the desired parameter.
    # ARG X : first time input (number of time points x 1).
    # ARG X2 : second time input (number of time points x 1)
    # ARG mode: indicates in which way the vectors X and X2 must be transposed
    # RETURN g : Gradient of the function with respect to the desired
    # parameter.
    #
    # Author : Tianqi Wei
    # Based on Matlab codes by David Luengo, 2007, 2008, Mauricio Alvarez, 2008

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
    # LFMGRADIENTH41 Gradient of the function h_i(z) with respect to some of the
    # hyperparameters of the kernel: m_k, C_k, D_k, m_r, C_r or D_r.
    # FORMAT
    # DESC Computes the gradient of the function h_i(z) with respect to some of
    # the parameters of the system (mass, spring or damper).
    # ARG gamma1 : Gamma value for first system.
    # ARG gamma2 : Gamma value for second system.
    # ARG sigma2 : length scale of latent process.
    # ARG gradThetaGamma : Vector with the gradient of gamma1 and gamma2 with
    # respect to the desired parameter.
    # ARG X : first time input (number of time points x 1).
    # ARG X2 : second time input (number of time points x 1)
    # ARG mode: indicates in which way the vectors X and X2 must be transposed
    # RETURN g : Gradient of the function with respect to the desired
    # parameter.
    #
    # Author : Tianqi Wei
    # Based on Matlab codes by David Luengo, 2007, 2008, Mauricio Alvarez, 2008

    if not mode:
        if not term:
            g = (gradUpsilon1 * gradThetaGamma) * (preExp/preFactor[0] - np.conj(preExp)/preFactor[1]) \
                + (compUpsilon1 * gradThetaGamma) * (- preExp/preFactorGrad[0] + np.conj(preExp)/preFactorGrad[1])
        else:
            g = (gradUpsilon1 * gradThetaGamma[0]) * (preExp/preFactor[0]) \
                + (compUpsilon1 * gradThetaGamma[0]) * (- preExp/preFactorGrad[0]) \
                + (np.conj(gradUpsilon1) * gradThetaGamma[1]) * (- preExp/preFactor[1]) \
                + (np.conj(compUpsilon1) * gradThetaGamma[1]) * (preExp/preFactorGrad[1])
    else:
        g = (gradUpsilon1 * gradThetaGamma[0]) *                                                                                  (preExp[:, 0]/preFactor[0] - preExp[:, 1]/preFactor[1]) \
            + (compUpsilon1 * gradThetaGamma[0]) * (- preExp[:, 0]/preFactorGrad[0] + preExp[:, 1]/preFactorGrad[1]) \
            + (gradUpsilon2 * gradThetaGamma[1]) * (preExp[:, 1]/preFactor[3] - preExp[:, 0]/preFactor[2]) \
            + (compUpsilon2 * gradThetaGamma[1]) * (- preExp[:, 1]/preFactorGrad[3] + preExp[:, 0]/preFactorGrad[2])
    return g.T

def lfmGradientH42(preFactor, preFactorGrad, gradThetaGamma, preExp, preExpt,
    compUpsilon1, compUpsilon2, mode, term=None):
    # LFMGRADIENTH42 Gradient of the function h_i(z) with respect to some of the
    # hyperparameters of the kernel: m_k, C_k, D_k, m_r, C_r or D_r.
    # FORMAT
    # DESC Computes the gradient of the function h_i(z) with respect to some of
    # the parameters of the system (mass, spring or damper).
    # ARG gamma1 : Gamma value for first system.
    # ARG gamma2 : Gamma value for second system.
    # ARG sigma2 : length scale of latent process.
    # ARG gradThetaGamma : Vector with the gradient of gamma1 and gamma2 with
    # respect to the desired parameter.
    # ARG X : first time input (number of time points x 1).
    # ARG X2 : second time input (number of time points x 1)
    # ARG mode: indicates in which way the vectors X and X2 must be transposed
    # RETURN g : Gradient of the function with respect to the desired
    # parameter.
    #
    # Author : Tianqi Wei
    # Based on Matlab codes by David Luengo, 2007, 2008, Mauricio Alvarez, 2008

    if not mode:
        if not term:
            g = compUpsilon1*(- (preExp/preFactorGrad[0] + preExpt/preFactor[0])*gradThetaGamma[0]
                + (np.conj(preExp)/preFactorGrad[1] + np.conj(preExpt)/preFactor[1])*gradThetaGamma[1])
        else:
            g = compUpsilon1*(- (preExp/preFactorGrad[0] + preExpt/preFactor[0])*gradThetaGamma) \
                + np.conj(compUpsilon1)*((preExp/preFactorGrad[1] + preExpt/preFactor[1])*gradThetaGamma)
    else:
        g = compUpsilon1*((preExp[:, 1]/preFactorGrad[2] + preExpt[:, 1]/preFactor[2])*gradThetaGamma[1]
            - (preExp[:, 0]/preFactorGrad[0] + preExpt[:, 0]/preFactor[0])*gradThetaGamma[0]) \
            - compUpsilon2*((preExp[:, 1]/preFactorGrad[3] + preExpt[:, 1]/preFactor[3])*gradThetaGamma[1]
            - (preExp[:, 0]/preFactorGrad[1] + preExpt[:, 0]/preFactor[1])*gradThetaGamma[0])
    return g.T

def lfmGradientSigmaH3(gamma1, gamma2, sigma2, X, X2, preFactor, mode, term=None):

    # LFMGRADIENTSIGMAH3 Gradient of the function h_i(z) with respect \sigma.
    # FORMAT
    # DESC Computes the gradient of the function h_i(z) with respect to the
    # length-scale of the input "force", \sigma.
    # ARG gamma1 : Gamma value for first system.
    # ARG gamma2 : Gamma value for second system.
    # ARG sigma2 : length scale of latent process.
    # ARG X : first time input (number of time points x 1).
    # ARG X2 : second time input (number of time points x 1).
    # ARG mode: indicates in which way the vectors X and X2 must be transposed
    # RETURN g : Gradient of the function with respect to \sigma.
    #
    # Author : Tianqi Wei
    # Based on Matlab codes by David Luengo, 2007, 2008, Mauricio Alvarez, 2008


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
    # LFMGRADIENTSIGMAH4 Gradient of the function h_i(z) with respect \sigma.
    # FORMAT
    # DESC Computes the gradient of the function h_i(z) with respect to the
    # length-scale of the input "force", \sigma.
    # ARG gamma1 : Gamma value for first system.
    # ARG gamma2 : Gamma value for second system.
    # ARG sigma2 : length scale of latent process.
    # ARG X : first time input (number of time points x 1).
    # ARG X2 : second time input (number of time points x 1).
    # ARG mode: indicates in which way the vectors X and X2 must be transposed
    # RETURN g : Gradient of the function with respect to \sigma.
    #
    # Author : Tianqi Wei
    # Based on Matlab codes by David Luengo, 2007, 2008, Mauricio Alvarez, 2008
    if not mode:
        if not term:
            g = lfmGradientSigmaUpsilonVector(gamma1, sigma2, X)*(preExp/preFactor[0] - np.conj(preExp)/preFactor[1]).T
        else:
            gradupsilon = lfmGradientSigmaUpsilonVector(gamma1, sigma2, X)
            g = (gradupsilon * (preExp/preFactor[0])).T - (np.conj(gradupsilon)*(preExp/preFactor[1])).T
    else:
        g = lfmGradientSigmaUpsilonVector(gamma1, sigma2, X)*(preExp[:, 0]/preFactor[0] - preExp[:, 1]/preFactor[1]).T \
            + lfmGradientSigmaUpsilonVector(gamma2, sigma2, X)*(preExp[:, 1]/preFactor[2] - preExp[:, 0]/preFactor[3]).T
    return g

    
def lfmDiagComputeH3(gamma, sigma2, t, factor, preExp, mode):
    upsi = lfmUpsilonVector(gamma ,sigma2, t)
    if mode:
        vec = preExp*upsi*factor
    else:
        temp = preExp*upsi
        vec = 2*np.real(temp/factor[0]) - temp/factor[1]
    return [vec, upsi] 


def lfmDiagComputeH4(gamma, sigma2, t, factor, preExp, mode):
    upsi = lfmUpsilonVector(gamma ,sigma2, t)
    if mode:
        vec = (preExp[:,0]/factor[0] -  2*preExp[:,1]/factor[1])*upsi
    else:
        temp2 = upsi*np.conj(preExp)/factor[1]
        vec = upsi*preExp/factor[0] - 2*np.real(temp2)
    return [vec, upsi]

def lfmDiagGradientH3(gamma, t, factor, preExp, compUpsilon, gradUpsilon, termH, preFactorGrad, gradTheta):
    expUpsilon = preExp*compUpsilon
    pgrad = - expUpsilon*(2/preFactorGrad**2)*gradTheta[0] - \
            (t*termH + preExp*gradUpsilon*factor[0] - expUpsilon*(1/gamma**2 - 2/preFactorGrad**2))*gradTheta[1]
    return pgrad

def lfmDiagGradientH4( t, factor, preExp, compUpsilon, gradUpsilon, gradTheta):
    pgrad = 2*preExp[:,1] * compUpsilon * (t/factor[1] + 1/factor[1]**2)*gradTheta[0] \
            + (gradUpsilon * (preExp[:,0]/factor[0] - 2*preExp[:,1]/factor[1])
            - compUpsilon * (t * preExp[:,0]/factor[0] + preExp[:,0]/factor[0]**2
            -  2*preExp[:,1]/factor[1]**2))*gradTheta[1]
    return pgrad

def lfmDiagGradientSH3(gamma, sigma2, t, factor, preExp, mode):
    upsi = lfmGradientSigmaUpsilonVector(gamma ,sigma2, t)
    if mode:
        vec = preExp * upsi*factor
    else:
        temp = preExp * upsi
        vec = 2*np.real(temp/factor[0]) - temp/factor[1]
    return [vec, upsi]

def lfmDiagGradientSH4(gamma, sigma2, t, factor, preExp, mode):
    upsi = lfmGradientSigmaUpsilonVector(gamma ,sigma2, t)
    if mode:
        vec = (preExp[:,0]/factor[0] -  2*preExp[:,1]/factor[1]) * upsi
    else:
        temp2 = upsi * np.conj(preExp)/factor[1]
        vec = upsi * preExp/factor[0] - 2*np.real(temp2)
    return [vec, upsi]

