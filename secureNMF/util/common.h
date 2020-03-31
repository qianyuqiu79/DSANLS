#ifndef COMMON_H_
#define COMMON_H_


#define DEBUG_MODE 0
#define DOUBLE_PRECISION false  // double or single prcesion floats
#define MAX_MATRIX_SIZE 1000000000

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <iostream>
#include <vector>

using namespace Eigen;

#if DEBUG_MODE
#define ASSERT(x) if (!(x)) \
	std::cout << "Assertion error in " << __FILE__ << " Line " << __LINE__ << std::endl
#else
#define ASSERT(x) 
#endif

#if DEBUG_MODE
#define check_success() \
	std::cout << "Complete " << __FILE__ << " Line " << __LINE__ << std::endl
#else
#define check_success() 
#endif

#if DEBUG_MODE
#define print_func() \
	std::cout << "Enter function \"" << __func__ << "\" in file " << __FILE__  << std::endl
#else
#define print_func() 
#endif

#if DEBUG_MODE
#define ERROR(x) \
    std::cout << "Error occurs in " << __FILE__ << " Line " << __LINE__ << ": " << x << std::endl
#else
#define ERROR(x) 
#endif

#if DOUBLE_PRECISION
#define FLOAT double
#define MPI_FLOAT_TYPE MPI_DOUBLE
#define DsMatrix MatrixXd
#define SpMatrix SparseMatrix<double>
#else
#define FLOAT float
#define MPI_FLOAT_TYPE MPI_FLOAT
#define DsMatrix MatrixXf
#define SpMatrix SparseMatrix<float>
#endif


inline void split_idx(const int num, const int size, std::vector<int> &splits, bool balanced) {
	if(balanced) {
		for (int rank = 0; rank < size; ++rank)
			splits[rank] = (num / size) * rank;
		splits[size] = num;
	}
	else{
		for (int rank = 0; rank < size; ++rank)
			splits[rank] = (num / 2 / size) * rank;
		splits[size] = num;
	}
}

#endif
