#include "common.h"



Matrix distribute_matrix(const Matrix M, const int num_row, const int num_col, const int *partition, LocalData *data)
{
    int size = data->size;
    int row_local = partition[data->rank + 1] - partition[data->rank];
    bool is_sparse;
    int *displacements = (int*)mkl_malloc((size + 1) * sizeof(int));
    int *counts = (int*)mkl_malloc(size * sizeof(int));

    if (data->rank == MPI_ROOT_RANK) {
        is_sparse = M.isSparse;
    }
    MPI_Barrier(data->comm);
    MPI_Bcast(&is_sparse, 1, MPI_BOOL, MPI_ROOT_RANK, data->comm);

    Matrix B;
    if (is_sparse) {
        int* new_rowIndex = NULL;
        if (data->rank == MPI_ROOT_RANK) {
            displacements[0] = 0;
            for (int i = 0; i < size; i++) {
                displacements[i + 1] = M.rowIndex[partition[i + 1]];
                counts[i] = displacements[i + 1] - displacements[i];
            }
            
            // adjust row index
            int offset = 0, rank = 0;
            new_rowIndex = (int*)mkl_malloc(M.numRow * sizeof(int));
            for (int i = 0; i < M.numRow; i++) {
                while (rank < data->size && i >= partition[rank + 1]) {
                    offset += counts[rank];
                    rank++;
                }
                new_rowIndex[i] = M.rowIndex[i] - offset;
            }
        }
        // send out values
        MPI_Barrier(data->comm);
        MPI_Bcast(displacements, size + 1, MPI_INT, MPI_ROOT_RANK, data->comm);
        MPI_Bcast(counts, size, MPI_INT, MPI_ROOT_RANK, data->comm);
        B = create_sparse_matrix(row_local, num_col, counts[data->rank]);
        MPI_Scatterv(M.values, counts, displacements, MPI_FLOAT_TYPE, B.values, counts[data->rank], MPI_FLOAT_TYPE, MPI_ROOT_RANK, data->comm);

        // send out columns
        MPI_Scatterv(M.columns, counts, displacements, MPI_INT, B.columns, counts[data->rank], MPI_INT, MPI_ROOT_RANK, data->comm);

        // send out rowIndex
        for (int i = 0; i < size; i++)
            counts[i] = partition[i + 1] - partition[i];
        MPI_Scatterv(new_rowIndex, counts, partition, MPI_INT, B.rowIndex, counts[data->rank], MPI_INT, MPI_ROOT_RANK, data->comm);
        B.rowIndex[B.numRow] = B.numNonzero;
        
        if (new_rowIndex)
            mkl_free(new_rowIndex);
    }
    // for dense matrix
    else {
        displacements[0] = 0;
        for (int i = 0; i < size; i++) {
            displacements[i + 1] = num_col * partition[i + 1];
            counts[i] = displacements[i + 1] - displacements[i];
        }
        B = create_dense_matrix(row_local, num_col);
        MPI_Barrier(data->comm);
        MPI_Scatterv(M.values, counts, displacements, MPI_FLOAT_TYPE, B.values, counts[data->rank], MPI_FLOAT_TYPE, MPI_ROOT_RANK, data->comm);
    }

    mkl_free(displacements);
    mkl_free(counts);
    return B;
}



// read M from files and distribute the data to other nodes
void read_and_distribute(const char *file_name, LocalData *data)
{
    int size = data->size;
    Matrix M, Mt;
    data->rowPartition = (int*)mkl_malloc((size + 1) * sizeof(int));
    data->colPartition = (int*)mkl_malloc((size + 1) * sizeof(int));

    if (data->rank == MPI_ROOT_RANK) {
        M = read_matrix(file_name);
        Mt = tranpose_matrix(M);

        int m_block = M.numRow / size;
        int n_block = M.numCol / size;
        int m_extra = M.numRow - m_block * size;
        int n_extra = M.numCol - n_block * size;

        data->colPartition[0] = 0;
        data->rowPartition[0] = 0;
        for (int i = 0; i < size; i++) {
            data->rowPartition[i + 1] = data->rowPartition[i] + m_block + ((i < m_extra) ? 1 : 0);
            data->colPartition[i + 1] = data->colPartition[i] + n_block + ((size - i <= n_extra) ? 1 : 0);
        }
    }
    MPI_Bcast(data->rowPartition, size + 1, MPI_INT, MPI_ROOT_RANK, data->comm);
    MPI_Bcast(data->colPartition, size + 1, MPI_INT, MPI_ROOT_RANK, data->comm);
    data->numRow = data->rowPartition[size];
    data->numCol = data->colPartition[size];

    Matrix MRow = distribute_matrix(M, data->numRow, data->numCol, data->rowPartition, data);
    Matrix MCol = distribute_matrix(Mt, data->numCol, data->numRow, data->colPartition, data);

    // convert to its tranpose
    data->MRow = tranpose_matrix(MRow);
    data->MCol = tranpose_matrix(MCol);
    destroy_matrix(&MRow);
    destroy_matrix(&MCol);

    if (data->rank == MPI_ROOT_RANK) {
        destroy_matrix(&M);
        destroy_matrix(&Mt);
    }
}