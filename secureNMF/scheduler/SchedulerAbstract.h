#ifndef NMF_SCHEDULERABSTRACT_H
#define NMF_SCHEDULERABSTRACT_H

#include "../util/base.h"
#include "../util/parameter.h"
#include "../util/monitor.h"
#include "../../matrix.h"

#define UPDATE_TAG 1
#define DUMP_TAG 2
#define TERMINATION_TAG 3

struct noprapere_tag {
};

class SchedulerAbstract {
protected:
    int world_size;
    int worker_rank;
    const Parameter &parameter;

    std::vector<int> n_partition;
    std::vector<int> m_partition;
    MatrixType M_part;
    int full_m;
    int full_n;
    int part_n;
    int part_m;
    DsMatrix U_local;
    DsMatrix V_part;
    DsMatrix U_part;

//    string dump_file;
//    ofstream dump_out;

    double local_norm;
    double total_norm;
    vector<double> local_loss_store;
    vector<double> local_time_store;
public:
    ///Do some common initialization

    SchedulerAbstract(const Parameter &parameter, noprapere_tag dummy) : parameter(parameter) {}

    explicit SchedulerAbstract(const Parameter &parameter) : parameter(parameter) {

        MPI_Init(NULL, NULL);
        MPI_Comm comm_world = MPI_COMM_WORLD;
        MPI_Comm_rank(MPI_COMM_WORLD, &worker_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        // Get the name of the processor
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int name_len;
        MPI_Get_processor_name(processor_name, &name_len);

        if (worker_rank == 0) {
            cout << parameter.get_all();
        }

        this->initialize_data();
//        dump_file = parameter.output_path+"_"+std::to_string(worker_rank)+".dump";
//        dump_out.open(dump_file.c_str());
//        if(!dump_out.good()){cerr<<"fail to open dump file\n"; exit(-1);}

//        dump_out<<std::setprecision(10);
    }

    virtual void initialize_data() {
        // worker_rank=0 -> server; also have data partition and work to do

        n_partition = vector<int>(world_size + 1, 0);

        if (!parameter.input_file.empty()) {
            M_part.read_matrix(parameter.input_file.c_str(), worker_rank, world_size, n_partition,
                               parameter.transpose_file_matrix, parameter.balanced);
        } else {
            // M_part.dense_random_matrix(100, 1000, rank, size, n_partition);
            M_part.sparse_test_matrix(1000, 1000, worker_rank, world_size, n_partition);
            //std::cout << "Need to provide filename of matrix." << std::endl;
            //return -1;
        }

        full_m = M_part.rows();
        full_n = n_partition[world_size];
        part_n = M_part.cols();

        U_local = DsMatrix(DsMatrix::Random(full_m, parameter.k).array() + 1.0) * (parameter.rand_upper_bound / 2);
        V_part = DsMatrix(DsMatrix::Random(part_n, parameter.k).array() + 1.0) * (parameter.rand_upper_bound / 2);

        m_partition = std::vector<int>(world_size + 1, 0);
        split_idx(full_m, world_size, m_partition,parameter.balanced);
        part_m = m_partition[worker_rank + 1] - m_partition[worker_rank];
        U_part = DsMatrix(U_local.block(m_partition[worker_rank], 0, part_m, parameter.k));

        local_norm = M_part.squaredNorm();
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Reduce(&local_norm, &total_norm, 1, MPI_DOUBLE, MPI_SUM, ROOT_RANK, MPI_COMM_WORLD);
    }

    virtual void calculate_NMF() = 0;

    virtual ~SchedulerAbstract() {
        MPI_Finalize();
    }

protected:
    // Coordinate descent for solving subproblem : min_X |A - X*B'|^2 + (mu/2)*|X-X_0|^2
    void coordinate_descent(DsMatrix &X, const MatrixType &A, const DsMatrix &B, const double mu) {
//        print_func();
        ASSERT(X.rows() == A.rows());
        ASSERT(X.cols() == B.cols());
        ASSERT(A.cols() == B.rows());

        const Index k = X.cols();
        for (Index i = 0; i < k; ++i) {
            X.col(i) = A * B.col(i) + mu * X.col(i);
            for (Index j = 0; j < k; ++j) {
                if (j != i) {
                    X.col(i) -= B.col(i).dot(B.col(j)) * X.col(j);
                }
            }
            auto norm = B.col(i).norm();
            X.col(i) = (X.col(i) / (norm * norm + mu)).cwiseMax(ZERO);
        }
    }


    double compute_loss(const MatrixType &M, const DsMatrix &U, const DsMatrix &V) {
        print_func();
        int cols_each_time = MAX_MATRIX_SIZE / M.rows();
        if (cols_each_time < 1)
            cols_each_time = 1;
        int col_ptr = 0;
        double ret = 0.0;
        while (col_ptr < M.cols()) {
            if (M.cols() - col_ptr < cols_each_time)
                cols_each_time = M.cols() - col_ptr;
            ret += (M.block(0, col_ptr, M.rows(), cols_each_time) -
                    U * V.block(col_ptr, 0, cols_each_time, V.cols()).transpose()).squaredNorm();
            col_ptr += cols_each_time;
        }

        return ret;
    }

    //calculate total loss for async algorithms
    void report_loss() {
        ofstream fout;
//        cout<<worker_rank<<endl;
        if (worker_rank == ROOT_RANK) {
            fout.open((parameter.output_path + "_time-loss.txt").c_str());
//            cout<<(parameter.output_path+"_time-loss.txt")<<endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Reduce(&local_norm, &total_norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        //read local loss for each dump iteration
        for (int epoch = 0; epoch <= parameter.max_epoch; ++epoch) {
            double total_loss = 0;
            double total_time = 0;
            double local_time = 0;
            double local_loss = 0;

            if (worker_rank) {
//                    ia>>local_time>>local_loss;
                local_time = local_time_store[epoch];
                local_loss = local_loss_store[epoch];
            }
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Reduce(&local_loss, &total_loss, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Reduce(&local_time, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            if (worker_rank == 0) {
                total_loss /= total_norm;
                total_time /= (world_size - 1);
                std::cout << "Iteration " << epoch << ": time=" << total_time << ", loss=" << total_loss << std::endl;
                fout << total_time << '\t' << total_loss << endl;
            }
        }

    }
};


#endif //NMF_SCHEDULERABSTRACT_H
