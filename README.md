# DSNALS: Distributed Sketched Non-Negative Matrix Factorization

This repo is an C implementation of the following paper:

> Yuqiu Qian, Conghui Tan, Nikos Mamoulis, David Cheung. *DSANLS: Accelerating Distributed Non-Negative Matrix
Factorization via Sketching*. WSDM 2018.

Please cite this paper if you use this code in your published research project.

## Prerequisite
This implementation requires:
- Message Passing Interface (MPI)
- Intel Math Kernel Library (MKL)

Before compilation, make sure the environment variable "MKLROOT" is properly set, and MPI can be found by your compiler.


## Usage

Basic usage:
```
./distNMF <input file name> k [optional arguments]
```

Optional arguments may be:
- -i \<number of iterations>
- -s \<row sketch ratio> \<column sketch ratio>, sketch ratio is a float between 0 - 1
- -o \<verbose frequency>
- -m \<sketch method>, choices include
  - 0 - Subsample
  - 1 - Gaussian
  - 2 - Uniform
- -g, use projected gradient descent to solve subproblem (default solver is regularized coordinate descent)
- -t \<alpha> \<beta>, where alpha and beta are parameters to decide the change rate of step sizes
  - For coordinate descent, mu = beta * iter ^ alpha
  - For gradient descent, eta = alpha / (1.0 + iter * beta)
- -u \<bound>, upper bound of the random numbers in the initalization of U and V


## Format of the input matrix:
The input matrix is a binary file and stored on the root node (the program will automatically distribute the data to other nodes).

### Desne matrix:
Dense matrix is grouped as follows:
- a byte of integer 0
- two 32-bits integers: m, n
- m\*n of float numbers, indicating the entries of the input matrix, stored in row-majored order

### Sparse matrix:
The sparse matrix is stored in [zero-based CSR format](https://software.intel.com/en-us/node/599835):
- a byte of integer 1
- three 32-bits integers: m, n, nnz, where nnz is the number of non-zero elements
- (m+1) of 32-bits integers, indicating the row index
- nnz of 32-bits integers: the column numbers of each non-zero entries
- nnz of 32-bits integers: the values of the non-zero entries

### Precision of float number
Precision of the floats (single-precision or double-precision) can be set by changing the macro "DOUBLE_PRECISION" in "common.h". Note that the precision of the float numbers in the input file must be consistent with this.


Updated by Yuqiu Qian
