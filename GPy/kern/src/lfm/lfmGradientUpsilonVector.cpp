/*
 * lfmGradientUpsilonVector.cpp 
 * 
 * Computes the Upsilon's Gradient wrt to gamma assuming t2 is the zero vector.
 * inputs:
 *  gamma: gamma value system
 *  sigma2: squared lengthscale
 *  t1: first time input (x 1)
 * outputs:
 *  gradient vector (x 1)
 *
 * This is a MEX-file for MATLAB.
 *
 * Diego Agudelo, 2015.
 *
 * How to compile this?
 *
 * 1) Generate the object file for Faddeeva.cc:
 *  > mex -c Faddeva.cc
 * 2) Compile the mex c++ file and link with the Faddeeva library
 *  > mex lfmGradientUpsilonVector.cpp  Faddeeva.o
*/
#include <iostream>
#include "Faddeeva.hh"
#include "mex.h"

#define PI  3.141592653589793

using namespace std;

bool isScalar(const mxArray *element) {
  mwSize rows = mxGetM(element);
  mwSize cols = mxGetN(element);
  return (rows == 1 && cols == 1);
}

bool isColumnVector(const mxArray *element) {
  mwSize cols = mxGetN(element);
  return (cols == 1);
}


void computeGradient(
    complex<double> gamma,
    double sigma2,
    double *t1,
    mwSize rows,
    double *gradientReal,
    double *gradientComplex) {
  complex<double> complexJ(0, 1);
  complex<double> complexMinusJ(0, -1);
  double sqrtpi = sqrt(PI);
  double dev = sqrt(sigma2);
  
  complex<double> Z2;
  complex<double> WOFZ2;

  Z2 = dev * gamma / 2.0;
  if (real(Z2) >= 0.0)
    WOFZ2 = Faddeeva::w(complexJ * Z2);
  else
    WOFZ2 = Faddeeva::w(complexMinusJ * Z2);
  
  for(mwSize i = 0; i < rows; i++) {
    double t1MinusT2 = t1[i]; // Since t2 is assumed to be the zero vector.
    complex<double> Z1 = t1MinusT2 / dev - dev * gamma / 2.0;
    complex<double> WOFZ1;
    if (real(Z1) >= 0.0)
      WOFZ1 = Faddeeva::w(complexJ * Z1);
    else
      WOFZ1 = Faddeeva::w(complexMinusJ * Z1);
    complex<double> ans;
    if (real(Z1) >= 0.0 && real(Z2) >= 0.0) {
      ans = (
        exp(sigma2 * gamma * gamma / 4.0 - gamma * t1MinusT2
        + log(sigma2 * gamma - 2.0 * t1MinusT2 ) )
        - exp( - (t1MinusT2 * t1MinusT2 / sigma2) + log(dev)
        + log(1.0/sqrtpi - Z1 * WOFZ1))
        + exp( - gamma * t1[i] + log(t1[i] * WOFZ2
        + dev * (1.0/sqrtpi - Z2 * WOFZ2)))
      );
    }
    if (real(Z1) < 0.0 &&  real(Z2) >= 0.0) {
      ans = (
        - exp(- (t1MinusT2 * t1MinusT2 / sigma2) + log(dev)
        + log(1.0/sqrtpi + Z1 * WOFZ1))
        + exp( - gamma * t1[i] + log( t1[i] * WOFZ2
        + dev*(1.0/sqrtpi - Z2 * WOFZ2)))
      );
    }
    if (real(Z1) >= 0.0 &&  real(Z2) < 0.0) {
      ans = (
        - exp( -(t1MinusT2 * t1MinusT2 / sigma2) + log(dev)
        + log(1/sqrtpi - Z1 * WOFZ1))
        - exp( - gamma * t1[i] + log(t1[i] * WOFZ2
        - dev * (1/sqrtpi + Z2 * WOFZ2)))
      );
    }
    if (real(Z1) < 0.0 &&  real(Z2) < 0.0) {
      ans = (
        - exp(sigma2 * gamma * gamma / 4.0 - gamma * t1MinusT2
        + log(sigma2 * gamma - 2.0 * t1MinusT2 ) )
        - exp( - (t1MinusT2 * t1MinusT2 / sigma2) + log(dev)
        + log(1.0/sqrtpi + Z1 * WOFZ1))
        - exp(- gamma * t1[i]
        + log(t1[i] * WOFZ2 - dev * (1.0/sqrtpi + Z2 * WOFZ2)))
      );
    }
    gradientReal[i] = real(ans);
    gradientComplex[i] = imag(ans);
  }
}
    

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // Input validation
  if (nrhs != 3)
    mexErrMsgTxt("Three inputs required.");
  if (!isScalar(prhs[0]))
    mexErrMsgTxt("Input #1 is not a scalar.");
  if (!isScalar(prhs[1]))
    mexErrMsgTxt("Input #2 is not a scalar.");
  if (!isColumnVector(prhs[2]))
    mexErrMsgTxt("Input #3 is not a column vector.");
  // Converting MATLAB variables to C++ variables
  double *gammaReal = mxGetPr(prhs[0]);
  double *gammaComplex = mxGetPi(prhs[0]);
  complex<double> gamma = gammaReal[0];
  if (mxIsComplex(prhs[0]))
    gamma = complex<double>(gammaReal[0], gammaComplex[0]);
  double sigma2 = mxGetScalar(prhs[1]);
  double *t1 = mxGetPr(prhs[2]);
  mwSize rows = mxGetM(prhs[2]);
  // Creating output matrix
  plhs[0] = mxCreateDoubleMatrix(rows, 1, mxCOMPLEX);
  double *gradientReal = mxGetPr(plhs[0]);
  double *gradientComplex = mxGetPi(plhs[0]);
  computeGradient(gamma, sigma2, t1, rows, gradientReal, gradientComplex);
}

