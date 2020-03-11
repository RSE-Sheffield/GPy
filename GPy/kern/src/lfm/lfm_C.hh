#include "complex"
#include "Python.h"
#include "npy_common.h"

static PyObject *UpsilonMatrix(PyObject *self, PyObject *args);

void C_UpsilonMatrix(
    std::complex<double> gamma,
    double sigma2,
    double *t1,
    double *t2,
    long rows,
    long cols,
    npy_cdouble *P_UpsilonMatrix_npy_data);

static PyObject *UpsilonVector(PyObject *self, PyObject *args) ;

void C_UpsilonVector(
    std::complex<double> gamma,
    double sigma2,
    double *t1,
    long rows,
    npy_cdouble *p_result);

static PyObject *GradientUpsilonMatrix(PyObject *self, PyObject *args);

void C_GradientUpsilonMatrix(
    std::complex<double> gamma,
    double sigma2,
    double *t1,
    double *t2,
    long rows,
    long cols,
    npy_cdouble *p_result);

static PyObject *GradientUpsilonVector(PyObject *self, PyObject *args) ;

void C_GradientUpsilonVector(
    std::complex<double> gamma,
    double sigma2,
    double *t1,
    long rows,
    npy_cdouble *p_result);

static PyObject *GradientSigmaUpsilonMatrix(PyObject *self, PyObject *args);

void C_GradientSigmaUpsilonMatrix(
    std::complex<double> gamma,
    double sigma2,
    double *t1,
    double *t2,
    long rows,
    long cols,
    npy_cdouble *p_result);

static PyObject *GradientSigmaUpsilonVector(PyObject *self, PyObject *args);

void C_GradientSigmaUpsilonVector(
    std::complex<double> gamma,
    double sigma2,
    double *t1,
    long rows,
    npy_cdouble * p_result);