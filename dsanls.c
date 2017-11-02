#include "timer.h"
#include "common.h"



// form sketched matrices: A'*S (or S'*A) and S'*B, 
// A can be either sparse or dense, B is dense
void sketch_matrix(const SketchMethod method, const int sketch_size, const int offset, const int local_size,
    const VSLStreamStatePtr stream, const Matrix A, const Matrix B, Matrix *SA, Matrix *SB)
{
    // Gaussian or uniform random sketching matrix
    if (method == GAUSSIAN || method == UNIFORM) {
        Matrix S = create_dense_matrix(A.numRow, sketch_size);
#if DEBUG_MODE
        for (int i = 0; i < S.numRow * S.numCol; i++)
            S.values[i] = 1.0;
#else
        if (method == GAUSSIAN)
            vRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, S.numRow * S.numCol, S.values, 0.0, 1.0);
        else
            vRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, S.numRow * S.numCol, S.values, -sqrt(3), sqrt(3));
#endif
        // SA = A' * S
        if (A.isSparse) {
            FLOAT one = 1.0;
            FLOAT zero = 0.0;
            mkl_csrmm("T", &(A.numRow), &(S.numCol), &(A.numCol), &one, "G  C", A.values, A.columns, 
                A.rowIndex, A.rowIndex + 1, S.values, &(S.numCol), &zero, SA->values, &sketch_size);
        }
        else
            cblas_gemm(CblasRowMajor, CblasTrans, CblasNoTrans, A.numCol, S.numCol, A.numRow,
                1.0, A.values, A.numCol, S.values, S.numCol, 0.0, SA->values, sketch_size);
        // SB = S' * B
        cblas_gemm(CblasRowMajor, CblasTrans, CblasNoTrans, S.numCol, B.numCol, local_size, 1.0,
            S.values + offset*S.numCol, S.numCol, B.values, B.numCol, 0.0, SB->values, SB->numCol);
        destroy_matrix(&S);
    }

    // Subsample sketching
    else if (method == SUBSAMPLE) {
        int *index_list = (int*)mkl_malloc(sketch_size * sizeof(int));

#if DEBUG_MODE
        for (int i = 0; i < sketch_size; i++)
            index_list[i] = i;
#else
        viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, sketch_size, index_list, 0, A.numRow);
#endif

        if (A.isSparse) {
            // SA = S' * A
            int offset = 0;
            for (int i = 0; i < sketch_size; i++) {
                int b = A.rowIndex[index_list[i]];
                int e = A.rowIndex[index_list[i] + 1];

                memcpy(SA->values + offset, A.values + b, (e - b) * sizeof(FLOAT));
                memcpy(SA->columns + offset, A.columns + b, (e - b) * sizeof(int));
                SA->rowIndex[i] = offset;
                offset += (e - b);
            }
            SA->numNonzero = offset;
            SA->rowIndex[sketch_size] = offset;
        }
        else {
            // SA = A' * S
            for (int i = 0; i < sketch_size; i++)
                cblas_copy(A.numCol, A.values + index_list[i] * A.numCol, 1, SA->values + i, sketch_size);
        }

        // form SB = S' * B
        FLOAT *zero_vector = (FLOAT*)mkl_malloc(B.numCol * sizeof(FLOAT));
        for (int i = 0; i < B.numCol; i++)
            zero_vector[i] = 0;

        for (int i = 0; i < sketch_size; i++)
            if (index_list[i] >= offset && index_list[i] < offset + local_size)
                cblas_copy(B.numCol, B.values + (index_list[i] - offset) * B.numCol, 1, SB->values + i * SB->numCol, 1);
            else
                cblas_copy(B.numCol, zero_vector, 1, SB->values + i * SB->numCol, 1);
        mkl_free(zero_vector);
        mkl_free(index_list);
    }
}




// gradient descent for solving: min |X*A' - op(B)|^2, s.t., X>=0
void gradient_descent(const int max_iter, const FLOAT step_size, const Matrix A, const Matrix B,
    const bool trans_B, Matrix X)
{
    Matrix tmp = create_dense_matrix(X.numRow, X.numCol);
    Matrix BA = create_dense_matrix(X.numRow, X.numCol);
    Matrix AtA = create_dense_matrix(A.numCol, A.numCol);

    // BA = op(B) * A
    if (B.isSparse) {
        FLOAT one = 1.0;
        FLOAT zero = 0.0;
        mkl_csrmm((trans_B) ? "T" : "N", &(B.numRow), &(BA.numCol), &(B.numCol), &one, "G  C", B.values,
            B.columns, B.rowIndex, B.rowIndex + 1, A.values, &(A.numCol), &zero, BA.values, &(BA.numCol));
    }
    else
        cblas_gemm(CblasRowMajor, (trans_B) ? CblasTrans : CblasNoTrans, CblasNoTrans, BA.numRow, BA.numCol, A.numRow, 1.0,
            B.values, B.numCol, A.values, A.numCol, 0.0, BA.values, BA.numCol);
    // AtA = A' * A
    cblas_syrk(CblasRowMajor, CblasUpper, CblasTrans, A.numCol, A.numRow, 1.0, A.values, A.numCol,
        0.0, AtA.values, AtA.numCol);

    // gradient descent
    for (int iter = 0; iter < max_iter; iter++) {
        // tmp = U * AtA
        cblas_symm(CblasRowMajor, CblasRight, CblasUpper, X.numRow, X.numCol, 1.0, AtA.values, AtA.numCol,
            X.values, X.numCol, 0.0, tmp.values, tmp.numCol);
        // X = X - step_size * tmp
        cblas_axpy(X.numRow * X.numCol, -step_size, tmp.values, 1, X.values, 1);
        // X = X + step_size * BA
        cblas_axpy(X.numRow * X.numCol, step_size, BA.values, 1, X.values, 1);

        nonnegative_projection(X);
    }

    destroy_matrix(&tmp);
    destroy_matrix(&BA);
    destroy_matrix(&AtA);
}




void coordinate_descent(const Matrix A, const Matrix B, const bool trans_B, Matrix X, const FLOAT mu)
{
    Matrix BA = create_dense_matrix(X.numRow, X.numCol);
    Matrix AtA = create_dense_matrix(A.numCol, A.numCol);

    // BA = op(B) * A
    if (B.isSparse) {
        FLOAT one = 1.0;
        FLOAT zero = 0.0;
        mkl_csrmm((trans_B)? "T": "N", &(B.numRow), &(BA.numCol), &(B.numCol), &one, "G  C", B.values,
            B.columns, B.rowIndex, B.rowIndex + 1, A.values, &(A.numCol), &zero, BA.values, &(BA.numCol));
    }
    else
        cblas_gemm(CblasRowMajor, (trans_B)? CblasTrans: CblasNoTrans, CblasNoTrans, BA.numRow, BA.numCol, A.numRow, 1.0,
            B.values, B.numCol, A.values, A.numCol, 0.0, BA.values, BA.numCol);
    if (mu > 0)
        cblas_axpy(X.numRow * X.numCol, mu, X.values, 1, BA.values, 1);

    // AtA = A' * A
    cblas_syrk(CblasRowMajor, CblasUpper, CblasTrans, A.numCol, A.numRow, 1.0, A.values, A.numCol,
        0.0, AtA.values, AtA.numCol);

    FLOAT *t = (FLOAT*)mkl_malloc(X.numCol * sizeof(FLOAT));

    // update column by column
    for (int i = 0; i < X.numCol; i++) {
        for (int j = 0; j < X.numCol; j++) {
            if (i == j)
                t[j] = 0;
            else if (i < j)
                t[j] = AtA.values[j + i * AtA.numCol];
            else
                t[j] = AtA.values[i + j * AtA.numCol];
        }

        cblas_gemv(CblasRowMajor, CblasNoTrans, X.numRow, X.numCol, -1.0, X.values, X.numCol,
            t, 1, 1.0, BA.values + i, BA.numCol);
        cblas_copy(X.numRow, BA.values + i, BA.numCol, X.values + i, X.numCol);
        cblas_scal(X.numRow, 1.0 / (AtA.values[i + i * AtA.numCol] + mu + EPSILON), X.values + i, X.numCol);

        for (int j = 0; j < X.numRow; j++)
            if (X.values[j*X.numCol + i] < 0)
                X.values[j*X.numCol + i] = 0;
    }

    mkl_free(t);

    destroy_matrix(&BA);
    destroy_matrix(&AtA);
}





void dsanls(const LocalData data, const SketchMethod method, const FLOAT row_ratio, const FLOAT col_ratio, const int max_iter, 
    const FLOAT alpha, const FLOAT beta, const bool use_gd, const int verbose, Matrix U, Matrix V)
{
    ASSERT_EQUAL(data.MRow.numCol, U.numRow);
    ASSERT_EQUAL(data.MCol.numCol, V.numRow);
    ASSERT_EQUAL(U.numCol, V.numCol);

    int k = U.numCol;

    int sketch_size_row = (int)ceil((double)data.numRow * row_ratio);
    int sketch_size_col = (int)ceil((double)data.numCol * col_ratio);

    Matrix SMRow, SMCol;
    if (data.MRow.isSparse && method == SUBSAMPLE)
        SMRow = create_sparse_matrix(sketch_size_col, data.MRow.numCol, data.MRow.numNonzero);
    else
        SMRow = create_dense_matrix(data.MRow.numCol, sketch_size_col);
    if (data.MCol.isSparse && method == SUBSAMPLE)
        SMCol = create_sparse_matrix(sketch_size_row, data.MCol.numCol, data.MCol.numNonzero);
    else
        SMCol = create_dense_matrix(data.MCol.numCol, sketch_size_row);

    Matrix SU = create_dense_matrix(sketch_size_row, k);
    Matrix SV = create_dense_matrix(sketch_size_col, k);
    Matrix SU_sum = create_dense_matrix(sketch_size_row, k);
    Matrix SV_sum = create_dense_matrix(sketch_size_col, k);

    // for gathering
    int *row_displs = NULL, *col_displs = NULL;
    int *row_counts = NULL, *col_counts = NULL;
    if (row_ratio == 1) {
        row_displs = (int*)mkl_malloc(data.size * sizeof(int));
        row_counts = (int*)mkl_malloc(data.size * sizeof(int));
        for (int i = 0; i < data.size; i++) {
            row_displs[i] = data.rowPartition[i] * k;
            row_counts[i] = (data.rowPartition[i + 1] - data.rowPartition[i]) * k;
        }
    }
    if (col_ratio == 1 || verbose > 0) {
        col_displs = (int*)mkl_malloc(data.size * sizeof(int));
        col_counts = (int*)mkl_malloc(data.size * sizeof(int));
        for (int i = 0; i < data.size; i++) {
            col_displs[i] = data.colPartition[i] * k;
            col_counts[i] = (data.colPartition[i + 1] - data.colPartition[i]) * k;
        }
    }

    // for verbose
    Matrix V_full;
    FLOAT square_norm_M;
    if (verbose > 0) {
        V_full = create_dense_matrix(data.numCol, k);
        FLOAT norm_local = cblas_nrm2(data.MRow.numNonzero, data.MRow.values, 1);
        norm_local *= norm_local;
        MPI_Allreduce(&norm_local, &square_norm_M, 1, MPI_FLOAT_TYPE, MPI_SUM, data.comm);

#if DEBUG_MODE
        if (data.rank == MPI_ROOT_RANK)
            printf("%e\n", square_norm_M);
#endif
    }

    // initialize random stream
    unsigned int random_seed;
    if (data.rank == MPI_ROOT_RANK)
        random_seed = (unsigned int)time(NULL);
    MPI_Barrier(data.comm);
    MPI_Bcast(&random_seed, 1, MPI_UNSIGNED, MPI_ROOT_RANK, data.comm);
    VSLStreamStatePtr random_stream;
    vslNewStream(&random_stream, VSL_BRNG_SFMT19937, random_seed);

    // for computing time
    double time_elapsed = 0.0;

    // main loop
    for (int iter = 0; iter < max_iter; iter++) {
        FLOAT step_size = alpha / (1.0 + iter * beta);
        FLOAT mu = alpha + iter * beta;

        // verbose
        if (verbose > 0 && iter % verbose == 0) {
            // gather complete V
            MPI_Barrier(data.comm);
            MPI_Allgatherv(V.values, V.numRow*k, MPI_FLOAT_TYPE, V_full.values, 
                col_counts, col_displs, MPI_FLOAT_TYPE, data.comm);
            FLOAT error = evaluate_factorization(data.MRow, V_full, U);
            FLOAT total_error;
            MPI_Reduce(&error, &total_error, 1, MPI_FLOAT_TYPE, MPI_SUM, MPI_ROOT_RANK, data.comm);
            if (data.rank == MPI_ROOT_RANK)
                printf("Iter: %d, Time: %f, Error: %e, Rel. Error: %e\n", iter, (double)time_elapsed, 
                    total_error, total_error / square_norm_M);
        }

        timer_start();

        // update U
        if (col_ratio < 1) {
            sketch_matrix(method, sketch_size_col, data.colPartition[data.rank], data.colPartition[data.rank + 1] - data.colPartition[data.rank],
                random_stream, data.MRow, V, &SMRow, &SV);
            MPI_Barrier(data.comm);
            MPI_Allreduce(SV.values, SV_sum.values, SV.numRow*SV.numCol, MPI_FLOAT_TYPE, MPI_SUM, data.comm);
        }
        else {
            MPI_Allgatherv(V.values, V.numRow*k, MPI_FLOAT_TYPE, SV_sum.values,
                col_counts, col_displs, MPI_FLOAT_TYPE, data.comm);
        }

        if (!use_gd) {
            if (col_ratio < 1)
                coordinate_descent(SV_sum, SMRow, SMRow.isSparse, U, mu);
            else
                coordinate_descent(SV_sum, data.MRow, true, U, 0);
        }
        else {
            if (col_ratio < 1)
                gradient_descent(GD_INNER_LOOP, step_size, SV_sum, SMRow, SMRow.isSparse, U);
            else
                gradient_descent(GD_INNER_LOOP, step_size, SV_sum, data.MRow, true, U);
        }


        // update V
        if (row_ratio < 1) {
            sketch_matrix(method, sketch_size_row, data.rowPartition[data.rank], data.rowPartition[data.rank + 1] - data.rowPartition[data.rank],
                random_stream, data.MCol, U, &SMCol, &SU);
            MPI_Barrier(data.comm);
            MPI_Allreduce(SU.values, SU_sum.values, SU.numRow*SU.numCol, MPI_FLOAT_TYPE, MPI_SUM, data.comm);
        }
        else {
            MPI_Allgatherv(U.values, U.numRow*k, MPI_FLOAT_TYPE, SU_sum.values,
                row_counts, row_displs, MPI_FLOAT_TYPE, data.comm);
        }

        if (!use_gd) {
            if (row_ratio < 1)
                coordinate_descent(SU_sum, SMCol, SMCol.isSparse, V, mu);
            else
                coordinate_descent(SU_sum, data.MCol, true, V, 0);
        } 
        else {
            if (row_ratio < 1)
                gradient_descent(GD_INNER_LOOP, step_size, SU_sum, SMCol, SMCol.isSparse, V);
            else
                gradient_descent(GD_INNER_LOOP, step_size, SU_sum, data.MCol, true, V);
        }

        time_elapsed += timer_stop();
    }

    destroy_matrix(&V_full);
    if (row_displs != NULL)
        mkl_free(row_displs);
    if (row_counts != NULL)
        mkl_free(row_counts);
    if (col_displs != NULL)
        mkl_free(col_displs);
    if (col_counts != NULL)
        mkl_free(col_counts);

    destroy_matrix(&SMRow);
    destroy_matrix(&SMCol);
    destroy_matrix(&SU);
    destroy_matrix(&SV);
    destroy_matrix(&SU_sum);
    destroy_matrix(&SV_sum);
}