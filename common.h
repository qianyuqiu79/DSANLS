#ifndef DSANLS_COMMON_H
#define DSANLS_COMMON_H

#define DOUBLE_PRECISION 0  // double or single prcesion floats
#define MAX_ITER 100
#define DEBUG_MODE 0
#define GD_INNER_LOOP 1     // number of steps of the gradient descent for each subproblem
#define EPSILON 1e-20
#define MAX_MEMORY_SIZE 4294967296   // maximum memory used when computing residual matrix for sparse data


typedef enum { SUBSAMPLE, GAUSSIAN, UNIFORM } SketchMethod;

#define max(x, y) ((x)>(y))?(x):(y)
#define min(x, y) ((x)<(y))?(x):(y)


#ifndef MKL_H
#include "mkl.h"
#define MKL_H
#endif


#ifndef OMP_H
#include "omp.h"
#define OMP_H
#endif


#ifndef MPI_H
#include "mpi.h"
#define MPI_H
#endif

#ifndef STDLIB_H
#include <stdlib.h>
#define STDLIB_H
#endif

#ifndef MATH_H
#include <math.h>
#define MATH_H
#endif

#ifndef STDIO_H
#include <stdio.h>
#define STDIO_H
#endif

#ifndef TIME_H
#include <time.h>
#define TIME_H
#endif

#ifndef STRING_H
#include<string.h>
#define STRING_H
#endif



#define ASSERT_EQUAL(x, y) \
    if ((x) != (y)) { \
        fprintf(stderr, "Error occurs in %s, line %d: %d != %d.\n", __FILE__, __LINE__, (x), (y)); \
        exit(-1); \
    }



#if DEBUG_MODE
#define ERROR(x) \
    { \
    fprintf(stderr, "Error occurs in %s, line %d: %s.\n", __FILE__, __LINE__, (x)); \
    getchar(); \
    exit(-1); \
    }


#define PRINT_MATRIX(A) \
    if ((A.isSparse)) { \
        fprintf(stderr, "\nSparse Matrix %s (%d*%d, NNZ %d, Line %d):\n", #A, A.numRow, A.numCol, A.numNonzero, __LINE__); \
        fprintf(stderr, "Row index: "); \
        for (int i = 0; i <= A.numRow && i <= 10; i++) \
            fprintf(stderr, "%d ", A.rowIndex[i]); \
        fprintf(stderr, "\nValues:  "); \
        for (int i = 0; i < A.numNonzero && i < 10; i++) \
            fprintf(stderr, "%.2e ", A.values[i]); \
        fprintf(stderr, "\nColumns: "); \
        for (int i = 0; i < A.numNonzero && i < 10; i++) \
            fprintf(stderr, "%8d ", A.columns[i]); \
        fprintf(stderr, "\n"); \
    \
    } \
    else { \
        fprintf(stderr, "\nMatrix %s (%d*%d, Line %d):\n", #A, A.numRow, A.numCol, __LINE__); \
        for (int i = 0; i < A.numRow && i < 5; i++) { \
            for (int j = 0; j < A.numCol && j < 5;j++) \
                fprintf(stderr, "%.2e ", A.values[i*A.numCol+j]); \
            fprintf(stderr, "\n"); \
            } \
        fprintf(stderr, "\n"); \
    }




#else
#define ERROR(x) 
#define PRINT_MATRIX(A) 
#endif



#ifndef bool
#define bool char
#define MPI_BOOL MPI_CHAR
#endif

#ifndef true
#define true 1
#endif

#ifndef false
#define false 0
#endif


#ifdef mkl_malloc
#undef mkl_malloc
#endif
#define mkl_malloc(size) MKL_malloc((size), 64)



#if DOUBLE_PRECISION
#define FLOAT double
#define MPI_FLOAT_TYPE MPI_DOUBLE
#define cblas_gemm cblas_dgemm
#define cblas_copy cblas_dcopy
#define cblas_axpy cblas_daxpy
#define cblas_nrm2 cblas_dnrm2
#define cblas_scal cblas_dscal 
#define cblas_syrk cblas_dsyrk
#define cblas_symm cblas_dsymm 
#define cblas_dot cblas_ddot
#define LAPACKE_potri LAPACKE_dpotri 
#define LAPACKE_potrf LAPACKE_dpotrf 
#define mkl_csrmm mkl_dcsrmm
#define vRngGaussian vdRngGaussian
#define vRngUniform vdRngUniform
#define vDiv vdDiv
#define vMul vdMul
#define mkl_omatcopy mkl_domatcopy 
#define mkl_imatcopy mkl_dimatcopy 
#define cblas_axpyi cblas_daxpyi
#define mkl_csrcsc mkl_dcsrcsc 
#define mkl_csrmultd mkl_dcsrmultd
#define cblas_gemv cblas_dgemv

#else
#define FLOAT float
#define MPI_FLOAT_TYPE MPI_FLOAT
#define cblas_gemm cblas_sgemm
#define cblas_copy cblas_scopy
#define cblas_axpy cblas_saxpy
#define cblas_nrm2 cblas_snrm2
#define cblas_scal cblas_sscal 
#define cblas_syrk cblas_ssyrk
#define cblas_symm cblas_ssymm 
#define cblas_dot cblas_sdot
#define LAPACKE_potri LAPACKE_spotri 
#define LAPACKE_potrf LAPACKE_spotrf 
#define mkl_csrmm mkl_scsrmm
#define vRngGaussian vsRngGaussian
#define vRngUniform vsRngUniform
#define vDiv vsDiv
#define vMul vsMul
#define mkl_omatcopy mkl_somatcopy 
#define mkl_imatcopy mkl_simatcopy
#define cblas_axpyi cblas_saxpyi
#define mkl_csrcsc mkl_scsrcsc 
#define mkl_csrmultd mkl_scsrmultd
#define cblas_gemv cblas_sgemv

#endif

typedef struct {
    bool isSparse;
    int numRow;
    int numCol;
    // number of non-zero elements
    int numNonzero;
    int *columns;
    // an index for speeding up locating, where rowIndex[j] storing the offset to the start of j-th row
    int *rowIndex;
    // values for non-zero elements
    FLOAT *values;
} Matrix;


Matrix create_dense_matrix(const int num_row, const int num_col);
Matrix create_sparse_matrix(const int num_row, const int num_col, const int num_nonzero);
void destroy_matrix(Matrix *A);
Matrix copy_matrix(const Matrix A);
Matrix read_matrix(const char *file_name);
Matrix tranpose_matrix(const Matrix A);
FLOAT evaluate_factorization(const Matrix M, const Matrix U, const Matrix V);
void nonnegative_projection(Matrix X);



#define MPI_ROOT_RANK 0

typedef struct {
    MPI_Comm comm;
    int size;
    int rank;
    int numRow;
    int numCol;
    int *rowPartition;
    int *colPartition;
    // the tranpose of matrix formed by the related rows of M
    Matrix MRow;
    // the matrix formed by the related columns of M
    Matrix MCol;
} LocalData;


void read_and_distribute(const char *file_name, LocalData *data);

void dsanls(const LocalData data, const SketchMethod method, const FLOAT row_ratio, const FLOAT col_ratio, const int max_iter,
    const FLOAT alpha, const FLOAT beta, const bool use_gd, const int verbose, Matrix U, Matrix V);


#endif
