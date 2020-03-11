/* This is a c file for accelerating the computation related to the latent force model kernal */

#include "Python.h"
#include "arrayobject.h"
#include "lfm_C.hh"
#include <math.h>
#include <iostream>
#include "Faddeeva.hh"

using namespace std;
#define MAX_COLS 10000
#define PI  3.141592653589793
/* #### Globals #################################### */

/* ==== Set up the methods table ====================== */
static PyMethodDef lfmMethods[] = {
	{"UpsilonMatrix", UpsilonMatrix, METH_VARARGS},
  {"UpsilonVector", UpsilonVector, METH_VARARGS},
  {"GradientUpsilonMatrix", GradientUpsilonMatrix, METH_VARARGS},
  {"GradientUpsilonVector", GradientUpsilonVector, METH_VARARGS},
  {"GradientSigmaUpsilonMatrix", GradientSigmaUpsilonMatrix, METH_VARARGS},
  {"GradientSigmaUpsilonVector", GradientSigmaUpsilonVector, METH_VARARGS},

  {NULL, NULL, 0, NULL}
	//{NULL, NULL}     /* Sentinel - marks the end of this structure */
};

//For python 2
// /* ==== Initialize the C_test functions ====================== */
// // Module name must be _C_arraytest in compile and linked 
// void init_lfm()  {
// 	Py_InitModule("lfm", lfmMethods);
// 	import_array();  // Must be present for NumPy.  Called first after above line.
// } 


/*For python 3 The method table must be referenced in the module definition structure:*/
static struct PyModuleDef lfm_C_module = {
    PyModuleDef_HEAD_INIT,
    "lfm_C",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    lfmMethods
};



PyMODINIT_FUNC
PyInit_lfm_C(void)
{
    PyObject * create = PyModule_Create(&lfm_C_module);
    import_array();
    return create;
}









/* #### Extensions ############################## */

/* 
 * Computes the Upsilon's Gradient wrt to gamma.
 * inputs:
 *  gamma: gamma value system
 *  sigma2: squared lengthscale
 *  t1: first time input (x 1)
 *  t2: second time input (y 1)
 * outputs:
 *  gradient Matrix (x y)
 */

static PyObject *UpsilonMatrix(PyObject *self, PyObject *args) 

{

    npy_cdouble gamma_npy;
    complex<double> gamma;
    double sigma2;
    PyArrayObject *t1, *t2;//, *UpsilonMatrix;

    long nrow, ncol;
    int UpsilonMatrix_dim[2];

	/* Parse tuples separately since args will differ between C fcns */
	  if (!PyArg_ParseTuple(args, "DdO!O!", 
		    &gamma_npy, &sigma2, &PyArray_Type, &t1, &PyArray_Type, &t2))  return NULL;
	  // if (gamma_npy == NULL )  
    //     return NULL;
    // else
        gamma = gamma_npy.real + gamma_npy.imag * 1i;  
    if (sigma2 == NULL )  return NULL;
    if (t1 == NULL )  return NULL;
    if (t2 == NULL )  return NULL;

	/* Get the dimensions of the input */
	nrow=UpsilonMatrix_dim[0] = t1->dimensions[0]; /* Get row dimension of t1*/
	ncol=UpsilonMatrix_dim[1] = t2->dimensions[0]; /* Get row dimension of t2*/
	cout<<"gamma = " << gamma_npy.real <<'+' <<gamma_npy.imag << 'i'<< "\n"<< "sigma2 = " << sigma2 << endl;

    /* Make a new double matrix of same dims */
  PyArrayObject * P_UpsilonMatrix_npy = (PyArrayObject * ) PyArray_FromDims(2, UpsilonMatrix_dim, NPY_CDOUBLE);
  C_UpsilonMatrix( gamma,sigma2, (double *) t1->data, (double *) t2->data, nrow, ncol, (npy_cdouble *) P_UpsilonMatrix_npy->data);

   return PyArray_Return(P_UpsilonMatrix_npy);
	
}

void C_UpsilonMatrix(
    std::complex<double> gamma,
    double sigma2,
    double *t1,
    double *t2,
    long rows,
    long cols,
    npy_cdouble *p_result) 
    
{

  cout<<"gamma="<<gamma<<" sigma2="<< sigma2 <<" rows="<<rows<<" cols="<<cols<<" "<< endl;
  for (int i=0;i<rows; i++)
  {
    cout<<"t1[" <<i<< "] = "<<t1[i]<<"\n";
  }
  for (int i=0;i<cols; i++)
  {
    cout<<"t2[" <<i<< "] = "<<t2[i]<<"\n";
  }
  complex<double> complexJ(0, 1);
  complex<double> complexMinusJ(0, -1);
  double dev = sqrt(sigma2);

  // assert(cols <= MAX_COLS);
  //complex<double> Z2[MAX_COLS];
  //complex<double> WOFZ2[MAX_COLS];
  
  complex<double> Z2[cols];
  complex<double> WOFZ2[cols];
  
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
      cout<<"ans = " << ans << endl;
      p_result[i*cols + j].real = real(ans);
      p_result[i*cols + j].imag = imag(ans);
    }
  }
}



 /* 
 * Computes Upsilon given a input vector
 * inputs:
 *  gamma: gamma value system
 *  sigma2: squared lengthscale
 *  t1: first time input (x 1)
 * outputs:
 *  upsilon vector (x 1)
 */
static PyObject *UpsilonVector(PyObject *self, PyObject *args) 

{

    npy_cdouble gamma_npy;
    complex<double> gamma;
    double sigma2;
    PyArrayObject *t1;
    long nrow;
    int UpsilonMatrix_dim[2];

	/* Parse tuples separately since args will differ between C fcns */
	  if (!PyArg_ParseTuple(args, "DdO!", 
		    &gamma_npy, &sigma2, &PyArray_Type, &t1))  return NULL;
	  // if (gamma_npy == NULL )  
    //     return NULL;
    // else
        gamma = gamma_npy.real + gamma_npy.imag * 1i;  
    if (sigma2 == NULL )  return NULL;
    if (t1 == NULL )  return NULL;

	/* Get the dimensions of the input */
	nrow=UpsilonMatrix_dim[0] = t1->dimensions[0]; /* Get row dimension of t1*/
	cout<<"gamma = " << gamma_npy.real <<'+' <<gamma_npy.imag << 'i'<< "\n"<< "sigma2 = " << sigma2 << endl;

    /* Make a new double matrix of same dims */
  PyArrayObject * P_UpsilonVector_npy = (PyArrayObject * ) PyArray_FromDims(1, UpsilonMatrix_dim, NPY_CDOUBLE);
  C_UpsilonVector( gamma,sigma2, (double *) t1->data, nrow, (npy_cdouble *) P_UpsilonVector_npy->data);

   return PyArray_Return(P_UpsilonVector_npy);
	
}


void C_UpsilonVector(
    complex<double> gamma,
    double sigma2,
    double *t1,
    long rows,
    npy_cdouble *p_result) 
{
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
  
  for(long i = 0; i < rows; i++) {
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
        2. * exp(sigma2 * gamma * gamma / 4.0 - gamma * t1MinusT2)
        - exp( - (t1MinusT2 * t1MinusT2 / sigma2) + log(WOFZ1))
        - exp( - gamma * t1[i] + log(WOFZ2))
      );
    }
    if (real(Z1) < 0.0 &&  real(Z2) >= 0.0) {
      ans = (
        exp(- (t1MinusT2 * t1MinusT2 / sigma2) + log(WOFZ1))
        - exp( - gamma * t1[i] + log(WOFZ2))
      );
    }
    if (real(Z1) >= 0.0 &&  real(Z2) < 0.0) {
      ans = (
        - exp(- (t1MinusT2 * t1MinusT2 / sigma2) + log(WOFZ1))
        + exp( - gamma * t1[i] + log(WOFZ2))
      );
    }
    if (real(Z1) < 0.0 &&  real(Z2) < 0.0) {
      ans = (
        -2. * exp(sigma2 * gamma * gamma / 4.0 - gamma * t1MinusT2)
        + exp( - (t1MinusT2 * t1MinusT2 / sigma2) + log(WOFZ1))
        + exp( - gamma * t1[i] + log(WOFZ2))
      );
    }
    p_result[i].real = real(ans);
    p_result[i].imag = imag(ans);
  }
}
    







/*
 * lfmGradientUpsilonMatrix.cpp 
 * 
 * Computes the Upsilon's Gradient wrt to gamma.
 * inputs:
 *  gamma: gamma value system
 *  sigma2: squared lengthscale
 *  t1: first time input (x 1)
 *  t2: second time input (y 1)
 * outputs:
 *  gradient Matrix (x y)
 */
static PyObject *GradientUpsilonMatrix(PyObject *self, PyObject *args) 

{

    npy_cdouble gamma_npy;
    complex<double> gamma;
    double sigma2;
    PyArrayObject *t1, *t2;//, *UpsilonMatrix;

    long nrow, ncol;
    int UpsilonMatrix_dim[2];

	/* Parse tuples separately since args will differ between C fcns */
	  if (!PyArg_ParseTuple(args, "DdO!O!", 
		    &gamma_npy, &sigma2, &PyArray_Type, &t1, &PyArray_Type, &t2))  return NULL;
	  // if (gamma_npy == NULL )  
    //     return NULL;
    // else
        gamma = gamma_npy.real + gamma_npy.imag * 1i;  
    if (sigma2 == NULL )  return NULL;
    if (t1 == NULL )  return NULL;
    if (t2 == NULL )  return NULL;

	/* Get the dimensions of the input */
	nrow=UpsilonMatrix_dim[0] = t1->dimensions[0]; /* Get row dimension of t1*/
	ncol=UpsilonMatrix_dim[1] = t2->dimensions[0]; /* Get row dimension of t2*/
	cout<<"gamma = " << gamma_npy.real <<'+' <<gamma_npy.imag << 'i'<< "\n"<< "sigma2 = " << sigma2 << endl;

    /* Make a new double matrix of same dims */
  PyArrayObject * P_GradientUpsilonMatrix_npy = (PyArrayObject * ) PyArray_FromDims(2, UpsilonMatrix_dim, NPY_CDOUBLE);
  C_GradientUpsilonMatrix( gamma,sigma2, (double *) t1->data, (double *) t2->data, nrow, ncol, (npy_cdouble *) P_GradientUpsilonMatrix_npy->data);

   return PyArray_Return(P_GradientUpsilonMatrix_npy);
	
}


void C_GradientUpsilonMatrix(
    complex<double> gamma,
    double sigma2,
    double *t1,
    double *t2,
    long rows,
    long cols,
    npy_cdouble *p_result)
 {
  complex<double> complexJ(0, 1);
  complex<double> complexMinusJ(0, -1);
  double sqrtpi = sqrt(PI);
  double dev = sqrt(sigma2);
  
  // assert(cols <= MAX_COLS);
  //complex<double> Z2[MAX_COLS];
  //complex<double> WOFZ2[MAX_COLS];
  
  complex<double> Z2[cols];
  complex<double> WOFZ2[cols];
  
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
          exp(sigma2 * gamma * gamma / 4.0 - gamma * t1MinusT2
          + log(sigma2 * gamma - 2.0 * t1MinusT2 ) )
          - exp( - (t1MinusT2 * t1MinusT2 / sigma2) + log(dev)
          + log(1.0/sqrtpi - Z1 * WOFZ1))
          + exp( - (t2[j] * t2[j] / sigma2) - gamma * t1[i]
          + log(t1[i] * WOFZ2[j]
          + dev * (1.0/sqrtpi - Z2[j] * WOFZ2[j])))
        );
      }
      if (real(Z1) < 0.0 &&  real(Z2[j]) >= 0.0) {
        ans = (
          - exp(- (t1MinusT2 * t1MinusT2 / sigma2) + log(dev)
          + log(1.0/sqrtpi + Z1 * WOFZ1))
          + exp( - (t2[j] * t2[j] / sigma2) - gamma * t1[i]
          + log( t1[i] * WOFZ2[j]
          + dev*(1.0/sqrtpi - Z2[j] * WOFZ2[j])))
        );
      }
      if (real(Z1) >= 0.0 &&  real(Z2[j]) < 0.0) {
        ans = (
          - exp( -(t1MinusT2 * t1MinusT2 / sigma2) + log(dev)
          + log(1/sqrtpi - Z1 * WOFZ1))
          - exp( -(t2[j] * t2[j] / sigma2) - gamma * t1[i]
          + log(t1[i] * WOFZ2[j]
          - dev * (1/sqrtpi + Z2[j] * WOFZ2[j])))
        );
      }
      if (real(Z1) < 0.0 &&  real(Z2[j]) < 0.0) {
        ans = (
          - exp(sigma2 * gamma * gamma / 4.0 - gamma * t1MinusT2
          + log(sigma2 * gamma - 2.0 * t1MinusT2 ) )
          - exp( - (t1MinusT2 * t1MinusT2 / sigma2) + log(dev)
          + log(1.0/sqrtpi + Z1 * WOFZ1))
          - exp( - (t2[j] * t2[j] / sigma2) - gamma * t1[i]
          + log(t1[i] * WOFZ2[j]
          - dev * (1.0/sqrtpi + Z2[j] * WOFZ2[j])))
        );
      }
      p_result[i*cols + j].real = real(ans);
      p_result[i*cols + j].imag = imag(ans);
    }
  }
}
    



/*
 * lfmGradientUpsilonVector.cpp 
 * 
 * Computes the Upsilon's Gradient wrt to Sigma assuming that t2 is zero vector.
 * inputs:
 *  gamma: gamma value system
 *  sigma2: squared lengthscale
 *  t1: first time input (x 1)
 * outputs:
 *  gradient vector (x 1)
*/
static PyObject *GradientUpsilonVector(PyObject *self, PyObject *args) 
{
    npy_cdouble gamma_npy;
    complex<double> gamma;
    double sigma2;
    PyArrayObject *t1;
    long nrow;
    int UpsilonMatrix_dim[2];

	/* Parse tuples separately since args will differ between C fcns */
	  if (!PyArg_ParseTuple(args, "DdO!", 
		    &gamma_npy, &sigma2, &PyArray_Type, &t1))  return NULL;
	  // if (gamma_npy == NULL )  
    //     return NULL;
    // else
        gamma = gamma_npy.real + gamma_npy.imag * 1i;  
    if (sigma2 == NULL )  return NULL;
    if (t1 == NULL )  return NULL;

	/* Get the dimensions of the input */
	nrow=UpsilonMatrix_dim[0] = t1->dimensions[0]; /* Get row dimension of t1*/
	cout<<"gamma = " << gamma_npy.real <<'+' <<gamma_npy.imag << 'i'<< "\n"<< "sigma2 = " << sigma2 << endl;

    /* Make a new double matrix of same dims */
  PyArrayObject * P_GradientUpsilonVector_npy = (PyArrayObject * )  PyArray_FromDims(1, UpsilonMatrix_dim, NPY_CDOUBLE);
  C_GradientUpsilonVector( gamma,sigma2, (double *) t1->data, nrow, (npy_cdouble *) P_GradientUpsilonVector_npy->data);

   return PyArray_Return(P_GradientUpsilonVector_npy);
}


void C_GradientUpsilonVector(
    complex<double> gamma,
    double sigma2,
    double *t1,
    long rows,
    npy_cdouble * p_result) 
    
  {
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
  
  for(long i = 0; i < rows; i++) {
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
    p_result[i].real = real(ans);
    p_result[i].imag = imag(ans);
  }
}









/*
 * lfmGradientSigmaUpsilonMatrix.cpp 
 * 
 * Computes the Upsilon's Gradient wrt to Sigma.
 * inputs:
 *  gamma: gamma value system
 *  sigma2: squared lengthscale
 *  t1: first time input (x 1)
 *  t2: second time input (y 1)
 * outputs:
 *  gradient Matrix (x y)
*/
static PyObject *GradientSigmaUpsilonMatrix(PyObject *self, PyObject *args) 

{

    npy_cdouble gamma_npy;
    complex<double> gamma;
    double sigma2;
    PyArrayObject *t1, *t2;//, *UpsilonMatrix;

    long nrow, ncol;
    int UpsilonMatrix_dim[2];

	/* Parse tuples separately since args will differ between C fcns */
	  if (!PyArg_ParseTuple(args, "DdO!O!", 
		    &gamma_npy, &sigma2, &PyArray_Type, &t1, &PyArray_Type, &t2))  return NULL;
	  // if (gamma_npy == NULL )  
    //     return NULL;
    // else
        gamma = gamma_npy.real + gamma_npy.imag * 1i;  
    if (sigma2 == NULL )  return NULL;
    if (t1 == NULL )  return NULL;
    if (t2 == NULL )  return NULL;

	/* Get the dimensions of the input */
	nrow=UpsilonMatrix_dim[0] = t1->dimensions[0]; /* Get row dimension of t1*/
	ncol=UpsilonMatrix_dim[1] = t2->dimensions[0]; /* Get row dimension of t2*/
	cout<<"gamma = " << gamma_npy.real <<'+' <<gamma_npy.imag << 'i'<< "\n"<< "sigma2 = " << sigma2 << endl;

    /* Make a new double matrix of same dims */
  PyArrayObject * P_GradientSigmaUpsilonMatrix_npy = (PyArrayObject * ) PyArray_FromDims(2, UpsilonMatrix_dim, NPY_CDOUBLE);
  C_GradientSigmaUpsilonMatrix( gamma,sigma2, (double *) t1->data, (double *) t2->data, nrow, ncol, (npy_cdouble *) P_GradientSigmaUpsilonMatrix_npy->data);

   return PyArray_Return(P_GradientSigmaUpsilonMatrix_npy);
	
}


void C_GradientSigmaUpsilonMatrix(
    complex<double> gamma,
    double sigma2,
    double *t1,
    double *t2,
    long rows,
    long cols,
    npy_cdouble * p_result)
{
  complex<double> complexJ(0, 1);
  complex<double> complexMinusJ(0, -1);
  double sqrtpi = sqrt(PI);
  double dev = sqrt(sigma2);
  
 // assert(cols <= MAX_COLS);
  //complex<double> Z2[MAX_COLS];
  //complex<double> WOFZ2[MAX_COLS];
  
  complex<double> Z2[cols];
  complex<double> WOFZ2[cols];
  
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
          dev * gamma * gamma * exp(sigma2 * gamma * gamma / 4.0 
          - gamma * t1MinusT2)
          - 2.0 * exp(- (t1MinusT2 * t1MinusT2 / sigma2)
          + log( t1MinusT2 * t1MinusT2 * WOFZ1 / (dev * dev * dev)
          + (t1MinusT2 / sigma2 + gamma / 2.0)
          * (1.0/sqrtpi - Z1 * WOFZ1) ) )
          - 2.0 * exp(-(t2[j] * t2[j] / sigma2) - gamma * t1[i]
          + log( t2[j] * t2[j] * WOFZ2[j] / (dev * dev * dev)
          + (t2[j] / sigma2 - gamma / 2.0)
          * (1.0 / sqrtpi - Z2[j] * WOFZ2[j]) ) )
        ); 
      }
      if (real(Z1) < 0.0 && real(Z2[j]) >= 0.0) {
        ans = (
          2.0 * exp( -(t1MinusT2 * t1MinusT2 / sigma2)
          + log( t1MinusT2 * t1MinusT2 * WOFZ1 / (dev * dev * dev)
          - ( t1MinusT2 / sigma2 + gamma / 2.0)
          * (1.0/sqrtpi + Z1 * WOFZ1) ) )
          - 2.0 * exp( -(t2[j] * t2[j] / sigma2) - gamma * t1[i]
          + log( t2[j] * t2[j] * WOFZ2[j] / (dev * dev * dev)
          + (t2[j]/sigma2 - gamma/2.0)
          * (1/sqrtpi - Z2[j]*WOFZ2[j]) ) )  
        );  
      }
      if (real(Z1) >= 0.0 && real(Z2[j]) < 0.0) {
        ans = (
          - 2.0 * exp( -(t1MinusT2 * t1MinusT2/ sigma2)
          + log( t1MinusT2 * t1MinusT2 * WOFZ1 / (dev * dev * dev)
          + ( t1MinusT2 / sigma2 + gamma / 2.0)
          * (1/sqrtpi - Z1 * WOFZ1) ) )
          + 2.0 * exp( - (t2[j] * t2[j] / sigma2) - gamma * t1[i]
          + log( t2[j] * t2[j] * WOFZ2[j] / (dev * dev * dev)
          - (t2[j]/sigma2 - gamma/2.0)
          * (1/sqrtpi + Z2[j] * WOFZ2[j])))
        );
      }
      if (real(Z1) < 0.0 && real(Z2[j]) < 0.0) {
        ans = (
          - dev * gamma * gamma * exp( sigma2 * gamma * gamma / 4.0
          - gamma * t1MinusT2)
          + 2.0 * exp( -( t1MinusT2 * t1MinusT2 / sigma2)
          + log( t1MinusT2 * t1MinusT2 * WOFZ1 / (dev  * dev * dev)
          - (t1MinusT2/sigma2 + gamma / 2.0)
          * (1.0/sqrtpi + Z1 * WOFZ1) ) )
          + 2.0 * exp( - (t2[j] * t2[j] / sigma2 - gamma * t1[i])
          + log( t2[j] * t2[j] * WOFZ2[j] / (dev * dev * dev)
          - (t2[j]/sigma2 - gamma/2.0)
          * (1.0/sqrtpi + Z2[j] * WOFZ2[j])))
        );
      }
      p_result[i*cols + j].real = real(ans);
      p_result[i*cols + j].imag = imag(ans);
    }
  }
}

    


/*
 * lfmGradientSigmaUpsilonVector.cpp
 * 
 * Computes the Upsilon's Gradient wrt to Sigma assuming that t2 is zero vector.
 * inputs:
 *  gamma: gamma value system
 *  sigma2: squared lengthscale
 *  t1: first time input (x 1)
 * outputs:
 *  gradient vector (x 1)
 */
static PyObject *GradientSigmaUpsilonVector(PyObject *self, PyObject *args) 
{
    npy_cdouble gamma_npy;
    complex<double> gamma;
    double sigma2;
    PyArrayObject *t1;
    long nrow;
    int UpsilonMatrix_dim[2];

	/* Parse tuples separately since args will differ between C fcns */
	  if (!PyArg_ParseTuple(args, "DdO!", 
		    &gamma_npy, &sigma2, &PyArray_Type, &t1))  return NULL;
	  // if (gamma_npy == NULL )  
    //     return NULL;
    // else
        gamma = gamma_npy.real + gamma_npy.imag * 1i;  
    if (sigma2 == NULL )  return NULL;
    if (t1 == NULL )  return NULL;

	/* Get the dimensions of the input */
	nrow=UpsilonMatrix_dim[0] = t1->dimensions[0]; /* Get row dimension of t1*/
	cout<<"gamma = " << gamma_npy.real <<'+' <<gamma_npy.imag << 'i'<< "\n"<< "sigma2 = " << sigma2 << endl;

    /* Make a new double matrix of same dims */
  PyArrayObject * P_GradientSigmaUpsilonVector_npy = (PyArrayObject * )  PyArray_FromDims(1, UpsilonMatrix_dim, NPY_CDOUBLE);
  C_GradientSigmaUpsilonVector( gamma,sigma2, (double *) t1->data, nrow, (npy_cdouble *) P_GradientSigmaUpsilonVector_npy->data);

   return PyArray_Return(P_GradientSigmaUpsilonVector_npy);
}

void C_GradientSigmaUpsilonVector(
    complex<double> gamma,
    double sigma2,
    double *t1,
    long rows,
    npy_cdouble * p_result)  
{
  complex<double> complexJ(0, 1);
  complex<double> complexMinusJ(0, -1);
  double sqrtpi = sqrt(PI);
  double dev = sqrt(sigma2);
  
  complex<double> Z2 = dev * gamma / 2.0;
  complex<double> WOFZ2;
  
  if (real(Z2) >= 0.0)
    WOFZ2 = Faddeeva::w(complexJ * Z2);
  else
    WOFZ2 = Faddeeva::w(complexMinusJ * Z2);
  
  for(long i = 0; i < rows; i++) {
    double t1MinusT2 = t1[i]; // Since t2 is assumed to be zero.
    complex<double> Z1 = t1MinusT2 / dev - dev * gamma / 2.0;
    complex<double> WOFZ1;
    if (real(Z1) >= 0.0)
      WOFZ1 = Faddeeva::w(complexJ * Z1);
    else
      WOFZ1 = Faddeeva::w(complexMinusJ * Z1);
    complex<double> ans;
    if (real(Z1) >= 0.0 && real(Z2) >= 0.0) {
      ans = (
        dev * gamma * gamma * exp(sigma2 * gamma * gamma / 4.0 
        - gamma * t1MinusT2)
        - 2.0 * exp(- (t1MinusT2 * t1MinusT2 / sigma2)
        + log( t1MinusT2 * t1MinusT2 * WOFZ1 / (dev * dev * dev)
        + (t1MinusT2 / sigma2 + gamma / 2.0)
        * (1.0/sqrtpi - Z1 * WOFZ1) ) )
        - 2.0 * exp( - gamma * t1[i]
        + log( (-gamma / 2.0)
        * (1.0 / sqrtpi - Z2 * WOFZ2) ) )
      ); 
    }
    if (real(Z1) < 0.0 && real(Z2) >= 0.0) {
      ans = (
        2.0 * exp( -(t1MinusT2 * t1MinusT2 / sigma2)
        + log( t1MinusT2 * t1MinusT2 * WOFZ1 / (dev * dev * dev)
        - ( t1MinusT2 / sigma2 + gamma / 2.0)
        * (1.0/sqrtpi + Z1 * WOFZ1) ) )
        - 2.0 * exp( - gamma * t1[i]
        + log( (- gamma/2.0) * (1/sqrtpi - Z2 * WOFZ2) ) )  
      );  
    }
    if (real(Z1) >= 0.0 && real(Z2) < 0.0) {
      ans = (
        - 2.0 * exp( -(t1MinusT2 * t1MinusT2/ sigma2)
        + log( t1MinusT2 * t1MinusT2 * WOFZ1 / (dev * dev * dev)
        + ( t1MinusT2 / sigma2 + gamma / 2.0)
        * (1/sqrtpi - Z1 * WOFZ1) ) )
        + 2.0 * exp( - gamma * t1[i]
        + log( (gamma/2.0) * (1/sqrtpi + Z2 * WOFZ2)))
      );
    }
    if (real(Z1) < 0.0 && real(Z2) < 0.0) {
      ans = (
        - dev * gamma * gamma * exp( sigma2 * gamma * gamma / 4.0
        - gamma * t1MinusT2)
        + 2.0 * exp( -( t1MinusT2 * t1MinusT2 / sigma2)
        + log( t1MinusT2 * t1MinusT2 * WOFZ1 / (dev  * dev * dev)
        - (t1MinusT2/sigma2 + gamma / 2.0)
        * (1.0/sqrtpi + Z1 * WOFZ1) ) )
        + 2.0 * exp( - gamma * t1[i]
        + log((gamma/2.0) * (1.0/sqrtpi + Z2 * WOFZ2)))
      );
    }
    p_result[i].real = real(ans);
    p_result[i].imag = imag(ans);
  }
}
    