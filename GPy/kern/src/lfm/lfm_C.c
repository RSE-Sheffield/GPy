/* This is a c file for accelerating the computation related to the latent force model kernal */

#include "Python.h"
#include "lfm.h"
#include <math.h>
//#include <stdio.h>
#include "Faddeeva.hh"

/* #### Globals #################################### */
extern "C"
/* ==== Set up the methods table ====================== */
static PyMethodDef lfmMethods[] = {
	{"computeUpsilonMatrix", computeUpsilonMatrix, METH_VARARGS},

  {NULL, NULL, 0, NULL}
	/*{NULL, NULL}     /* Sentinel - marks the end of this structure */
};

/* ==== Initialize the C_test functions ====================== */
// Module name must be _C_arraytest in compile and linked 
void init_lfm()  {
	(void) Py_InitModule("lfm", lfmMethods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}


/* #### Extensions ############################## */

static PyObject *computeUpsilonMatrix(PyObject *self, PyObject *args) {

    npy_cdouble gamma_npy;
    complex<double> gamma;
    double sigma2;
    PyArrayObject *t1, *t2, *UpsilonMatrix;

    long nrow_t1, ncol_t1, nrow_t2, ncol_t2, UpsilonMatrix_dim[2]

	/* Parse tuples separately since args will differ between C fcns */
	  if (!PyArg_ParseTuple(args, "DdO!O!", 
		    &gamma_npy, &sigma2, &PyArray_Type, &t1, &PyArray_Type, &t2))  return NULL;
	  if (gamma_npy == NULL )  
        return NULL;
    else
        gamma = gamma_npy.real + gamma_npy.imag * i;  
    if (sigma2 == NULL )  return NULL;
    if (t1 == NULL )  return NULL;
    if (t2 == NULL )  return NULL;

	/* Get the dimensions of the input */
	nrow=UpsilonMatrix_dim[0] = t1->dimensions[0]; /* Get row dimension of t1*/
	ncol=UpsilonMatrix_dim[1] = t2->dimensions[0]; /* Get row dimension of t2*/

    /* Make a new double matrix of same dims */
  P_UpsilonMatrix_npy = (PyArrayObject *) PyArray_FromDims(2, UpsilonMatrix_dim,NPY_CDOUBLE);
  C_computeUpsilonMatrix( gamma,sigma2, t1, t2, nrow, ncol, *(P_UpsilonMatrix_npy.data));
 	
	
}

void C_computeUpsilonMatrix(
    complex<double> gamma,
    double sigma2,
    double *t1,
    double *t2,
    long rows,
    long cols,
    npy_cdouble *P_UpsilonMatrix_npy_data,) {
  complex<double> complexJ(0, 1);
  complex<double> complexMinusJ(0, -1);
  double dev = sqrt(sigma2);

  // assert(cols <= MAX_COLS);
  complex<double> Z2[MAX_COLS];
  complex<double> WOFZ2[MAX_COLS];
  
  // complex<double> Z2[cols];
  // complex<double> WOFZ2[cols];
  
  for(long j = 0; j < cols; j++) {
    Z2[j] = t2[j] / dev + dev * gamma / 2.0;
    if (real(Z2[j]) >= 0.0)
      WOFZ2[j] = Faddeeva::w(complexJ * Z2[j]);
    else
      WOFZ2[j] = Faddeeva::w(complexMinusJ * Z2[j]);
  }
  
  for(long i = 0; i < rows; i++) {
    for(long j = 0; j < cols; j++) {
      double t1MinusT2 = t1[i] - t2[j];
      complex<double> Z1 = t1MinusT2 / dev - dev * gamma / 2.0;
      complex<double> WOFZ1;
      if (real(Z1) >= 0.0)
        WOFZ1 = Faddeeva::w(complexJ * Z1);
      else
        WOFZ1 = Faddeeva::w(complexMinusJ * Z1);
      complex<double> ans;
      if (real(Z1) >= 0.0 && real(Z2[j]) >= 0.0) {
        ans = (
          2.0 * exp(sigma2 * (gamma*gamma)/4.0 - gamma * t1MinusT2)
          - exp( -(t1MinusT2 * t1MinusT2 / sigma2) + log(WOFZ1))
          - exp( -(t2[j] * t2[j] / sigma2) - gamma * t1[i] + 
          log(WOFZ2[j]) )
        ); 
      }
      if (real(Z1) < 0.0 && real(Z2[j]) >= 0.0) {
        ans = (
          exp( -(t1MinusT2 * t1MinusT2 / sigma2) + log(WOFZ1))
          - exp( -(t2[j] * t2[j] / sigma2) - gamma * t1[i] + 
          log(WOFZ2[j]) )
        );  
      }
      if (real(Z1) >= 0.0 && real(Z2[j]) < 0.0) {
        ans = (
          - exp( -(t1MinusT2 * t1MinusT2 / sigma2) + log(WOFZ1))
          + exp( -(t2[j] * t2[j] / sigma2) - gamma * t1[i] + 
          log(WOFZ2[j]) )
        );
      }
      if (real(Z1) < 0.0 && real(Z2[j]) < 0.0) {
        ans = (
          -2.0 * exp(sigma2 * (gamma*gamma)/4.0 - gamma * t1MinusT2)
          + exp( -(t1MinusT2 * t1MinusT2 / sigma2) + log(WOFZ1))
          + exp( -(t2[j] * t2[j] / sigma2) - gamma * t1[i] + 
          log(WOFZ2[j]) )
        );
      }
      P_UpsilonMatrix_npy_data[i + rows * j].real = real(ans);
      P_UpsilonMatrix_npy_data[i + rows * j].imag = imag(ans);
    }
  }
}
