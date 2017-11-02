#include "common.h"


Matrix create_dense_matrix(const int num_row, const int num_col) {
    Matrix A;
    A.isSparse = false;
    A.numRow = num_row;
    A.numCol = num_col;
    A.numNonzero = num_row * num_col;
    A.values = (FLOAT*)mkl_malloc(num_row*num_col * sizeof(FLOAT));
    A.rowIndex = NULL;
    A.columns = NULL;
    return A;
}


Matrix create_sparse_matrix(const int num_row, const int num_col, const int num_nonzero) {
    Matrix A;
    A.isSparse = true;
    A.numRow = num_row;
    A.numCol = num_col;
    A.numNonzero = num_nonzero;
    A.values = (FLOAT*)mkl_malloc(num_nonzero * sizeof(FLOAT));
    A.columns = (int*)mkl_malloc(num_nonzero * sizeof(int));
    A.rowIndex = (int*)mkl_malloc((num_row + 1) * sizeof(int));
    return A;
}


void destroy_matrix(Matrix *A) {
    A->numRow = 0;
    A->numCol = 0;
    A->numNonzero = 0;
    if (A->values != NULL)
        mkl_free(A->values);
    A->values = NULL;
    if (A->isSparse) {
        mkl_free(A->columns);
        A->columns = NULL;
        mkl_free(A->rowIndex);
        A->rowIndex = NULL;
    }
}


Matrix copy_matrix(const Matrix A) {
    Matrix B;
    if (A.isSparse) {
        B = create_sparse_matrix(A.numRow, A.numCol, A.numNonzero);
        cblas_copy(A.numNonzero, A.values, 1, B.values, 1);
        memcpy(B.columns, A.columns, A.numNonzero * sizeof(int));
        memcpy(B.rowIndex, A.rowIndex, (A.numRow + 1) * sizeof(int));
    }
    else {
        B = create_dense_matrix(A.numRow, A.numCol);
        cblas_copy(A.numRow*A.numCol, A.values, 1, B.values, 1);
    }
    return B;
}


// read the matrix from data
Matrix read_matrix(const char *file_name) {
    FILE *fp = fopen(file_name, "rb");
    if (fp == NULL)
        ERROR("input stream is not accessible");
    bool is_sparse = 0;
    int m, n;
    fread(&is_sparse, sizeof(bool), 1, fp);
    fread(&m, sizeof(int), 1, fp);
    fread(&n, sizeof(int), 1, fp);

    Matrix A;
    if (is_sparse) {
        int nnz;  // number of non-zeros elements
        fread(&nnz, sizeof(int), 1, fp);

        A = create_sparse_matrix(m, n, nnz);
        fread(A.rowIndex, sizeof(int), m + 1, fp);
        fread(A.columns, sizeof(int), nnz, fp);
        fread(A.values, sizeof(FLOAT), nnz, fp);
    }
    else {
        A = create_dense_matrix(m, n);
        fread(A.values, sizeof(FLOAT), m*n, fp);
    }
    fclose(fp);
    return A;
}




// return the tranpose of A
Matrix tranpose_matrix(const Matrix A)
{
    Matrix B;
    if (A.isSparse) {
        int n = max(A.numRow, A.numCol);
        int *old_rowIndex;
        if (A.numRow < n) {
            old_rowIndex = (int*)mkl_malloc((n + 1) * sizeof(int));
            memcpy(old_rowIndex, A.rowIndex, (A.numRow + 1) * sizeof(int));
            for (int i = A.numRow + 1; i <= n; i++)
                old_rowIndex[i] = A.rowIndex[A.numRow];
        }
        else
            old_rowIndex = A.rowIndex;

        int job[] = { 0, 0, 0, 0, 0, 1 };
        int info;

        Matrix B = create_sparse_matrix(A.numCol, A.numRow, A.numNonzero);
        int *new_rowIndex;
        if (B.numRow < n)
            new_rowIndex = (int*)mkl_malloc((n + 1) * sizeof(int));
        else
            new_rowIndex = B.rowIndex;

        mkl_csrcsc(job, &n, A.values, A.columns, old_rowIndex, B.values, B.columns, new_rowIndex, &info);

        if (new_rowIndex != B.rowIndex) {
            memcpy(B.rowIndex, new_rowIndex, (B.numRow + 1) * sizeof(int));
            mkl_free(new_rowIndex);
        }
        if (old_rowIndex != A.rowIndex)
            mkl_free(old_rowIndex);

        return B;
    }
    else {
        B = create_dense_matrix(A.numCol, A.numRow);
        mkl_omatcopy('R', 'T', A.numRow, A.numCol, 1.0, A.values, A.numCol, B.values, B.numCol);
    }

    return B;
}



// return the total error of the factorization M = U*V'
FLOAT evaluate_factorization(const Matrix M, const Matrix U, const Matrix V)
{
    ASSERT_EQUAL(M.numRow, U.numRow);
    ASSERT_EQUAL(M.numCol, V.numRow);
    ASSERT_EQUAL(U.numCol, V.numCol);

    static size_t memory_size = MAX_MEMORY_SIZE;

    FLOAT error = 0;
    int row = 0;

    if (M.isSparse) {
        FLOAT *tmp = NULL;
        int numRow = min(M.numRow, memory_size / (M.numCol * sizeof(FLOAT)));
        tmp = (FLOAT*)mkl_malloc(numRow * M.numCol * sizeof(FLOAT));

        // try to find a suitable memory size
        while (tmp == NULL) {
            numRow = numRow / 2 + (numRow % 2);
            tmp = (FLOAT*)mkl_malloc(numRow * M.numCol * sizeof(FLOAT));
        }
        memory_size = numRow * M.numCol * sizeof(FLOAT);

        int offset = 0;
        while (offset < M.numRow) {
            if (offset + numRow > M.numRow)
                numRow = M.numRow - offset;

            cblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans, numRow, M.numCol, U.numCol, 1.0, 
                U.values + offset * U.numCol, U.numCol, V.values, V.numCol, 0.0, tmp, M.numCol);

            for (int i = 0; i < numRow; i++) {
                int row_id = i + offset;
                cblas_axpyi(M.rowIndex[row_id + 1] - M.rowIndex[row_id], -1.0, M.values + M.rowIndex[row_id],
                    M.columns + M.rowIndex[row_id], tmp + i * M.numCol);
            }

            FLOAT norm = cblas_nrm2(numRow * M.numCol, tmp, 1);
            error += norm * norm;

            offset += numRow;
        }

        mkl_free(tmp);
    }

    else {
        Matrix tmp = create_dense_matrix(M.numRow, M.numCol);

        // tmp = U * V'
        cblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans, M.numRow, M.numCol, U.numCol, 1.0, U.values, U.numCol,
            V.values, V.numCol, 0.0, tmp.values, tmp.numCol);
        // tmp -= M
        cblas_axpy(M.numRow * M.numCol, -1.0, M.values, 1, tmp.values, 1);
        error = cblas_nrm2(M.numRow * M.numCol, tmp.values, 1);
        error *= error;
        destroy_matrix(&tmp);
    }

    return error;
}



void nonnegative_projection(Matrix X)
{
    for (int i = 0; i < X.numNonzero; i++)
        if (X.values[i] < 0)
            X.values[i] = 0;
}