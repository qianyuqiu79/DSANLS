#ifndef MATRIX_H_
#define MATRIX_H_
#include"common.h"

#define ZERO 0

class MatrixType {
private:
	bool isSparse_;
	DsMatrix denseM_;
	SpMatrix sparseM_;

public:

	MatrixType() = default;

	MatrixType(const MatrixType&) = default;

	MatrixType(const DsMatrix& denseM){ 
		isSparse_ = false;
		denseM_ = denseM;
	}

	MatrixType(const SpMatrix& sparseM) {
		isSparse_ = true;
		sparseM_ = sparseM; 
	}

	MatrixType(DsMatrix&& denseM) noexcept {
		isSparse_ = false;
		denseM_ = std::move(denseM);
	}

	MatrixType(SpMatrix&& sparseM) noexcept {
		isSparse_ = true;
		sparseM_ = std::move(sparseM);
	}

	bool isSparse() const {
		return isSparse_;
	}

	const DsMatrix& dense() const {
		ASSERT(isSparse_ == false);
		return denseM_;
	}

	const SpMatrix& sparse() const {
		ASSERT(isSparse_ == true);
		return sparseM_;
	}

	DsMatrix& dense() {
		ASSERT(isSparse_ == false);
		return denseM_;
	}

	SpMatrix& sparse() {
		ASSERT(isSparse_ == true);
		return sparseM_;
	}

	Index cols() const {
		if (isSparse_) return sparseM_.cols();
		else return denseM_.cols();
	}

	MatrixType col(const Index i) const {
		if (isSparse_) {
			SpMatrix A = sparseM_.col(i);
			return MatrixType(A);
		}
		else {
			DsMatrix B = denseM_.col(i);
			return MatrixType(B);
		}
	}

	Index rows() const  {
		if (isSparse_) {
			return sparseM_.rows();
		}
		else { 
			return denseM_.rows();
		}
	}

	MatrixType row(const Index i) const {
		if (isSparse_) {
			SpMatrix A = sparseM_.row(i);
			return MatrixType(A);
		}
		else {
			DsMatrix B = denseM_.row(i);
			return MatrixType(B);
		}
	}

	/*
	void set_col(const Index i, const MatrixType &A) {
		if (isSparse_) {
			// sparseM_.col(i) = A.sparseM_;
		}
		else {
			denseM_.col(i) = A.denseM_;
		}
	}
	void set_row(const Index i, const MatrixType &A) {
		if (isSparse_) {
			// sparseM_.row(i) = A.sparseM_;
		}
		else {
			denseM_.row(i) = A.denseM_;
		}
	}
	*/

	MatrixType block(const int i, const int j, const int p, const int q) const {
		if (isSparse_) {
			SpMatrix A = sparseM_.block(i, j, p, q);
			return MatrixType(std::move(A));
		}
		else { 
			DsMatrix A = denseM_.block(i, j, p, q);
			return MatrixType(std::move(A));
		}
	}

	MatrixType transpose() const {
		if (isSparse_) {
			SpMatrix A = sparseM_.transpose();
			return MatrixType(std::move(A));
		}
		else {
			DsMatrix A = denseM_.transpose();
			return MatrixType(std::move(A));
		}
	}

	double squaredNorm() const{
		if (isSparse_) {
			return sparseM_.squaredNorm();
		}
		else {
			return denseM_.squaredNorm();
		}
	}


	DsMatrix operator*(const DsMatrix &A) const {
		if (isSparse_) {
			return sparseM_ * A;
		}
		else {
			return denseM_ * A;
		}
	}

	DsMatrix operator-(const DsMatrix &A) const {
		if (isSparse_) {
			return sparseM_ - A;
		}
		else {
			return denseM_ - A;
		}
	}

	MatrixType& operator=(MatrixType &&A) noexcept {
		isSparse_ = A.isSparse_;
		if (isSparse_) {
			sparseM_ = std::move(A.sparseM_);
		}
		else {
			denseM_ = std::move(A.denseM_);
		}
		return *this;
	}

	MatrixType& operator=(const MatrixType &A) {
		isSparse_ = A.isSparse_;
		if (isSparse_) {
			sparseM_ = A.sparseM_;
		}
		else {
			denseM_ = A.denseM_;
		}
		return *this;
	}
	
	void read_matrix(const char *file_name, const int rank, const int size,
		std::vector<int> &n_splits, const bool transpose = false, const bool balanced = true);

	void dense_random_matrix(const int full_m, const int full_n, const int rank, const int size, std::vector<int> &n_splits);
	void sparse_test_matrix(const int full_m, const int full_n, const int rank, const int size, std::vector<int> &n_splits);
	
};

#endif
