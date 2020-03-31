#ifndef NMF_SYNCSKTUV_H
#define NMF_SYNCSKTUV_H

#include "SchedulerAbstract.h"

class SyncSkt_UV : public SchedulerAbstract{
public:
    SyncSkt_UV(const Parameter& parameter): SchedulerAbstract(parameter){

    }

    void calculate_NMF(){
        ofstream fout;
//        cout<<worker_rank<<endl;
        if(worker_rank==ROOT_RANK){
            fout.open((parameter.output_path+"_time-loss.txt").c_str());
//            cout<<(parameter.output_path+"_time-loss.txt")<<endl;
        }
        Monitor timer;
        double time_elapsed = 0;
        double total_time_elapsed=0;
        const int d_v = int(part_n * parameter.col_ratio);//part_n * parameter.col_ratio
        const int d_u = int(full_m * parameter.row_ratio);//full_m * parameter.row_ratio

        DsMatrix U_global_sum = DsMatrix(full_m, parameter.k);

        MatrixType M_skt_V;// = DsMatrix(full_m,d);
        MatrixType M_skt_U; // =DsMatrix(part_n,d);

        std::vector<std::vector<std::pair<Index, FLOAT>>> sparserowIndex;
        std::vector<std::vector<std::pair<Index, FLOAT>>> sparsecolIndex;

        if (M_part.isSparse()) {
            sparserowIndex = std::vector<std::vector<std::pair<Index, FLOAT>>>(M_part.rows(), std::vector<std::pair<Index, FLOAT>>());
            sparsecolIndex = std::vector<std::vector<std::pair<Index, FLOAT>>>(M_part.cols(), std::vector<std::pair<Index, FLOAT>>());

            M_skt_U = MatrixType(SpMatrix(part_n, d_u));
            M_skt_V = MatrixType(SpMatrix(full_m, d_v));
            // M_part.sparse().makeCompressed();
            int col = 0;
            int row;
            int count = 0;
            for (int idx = 0; idx < M_part.sparse().innerSize(); ++idx) {
                while (count == M_part.sparse().outerIndexPtr()[col + 1]) {
                    col += 1;
                }
                row = M_part.sparse().innerIndexPtr()[idx];
                sparserowIndex[row].push_back(std::make_pair(col, M_part.sparse().valuePtr()[idx]));
                count += 1;
            }

            col = 0;
            count = 0;
            for (int idx = 0; idx < M_part.sparse().innerSize(); ++idx) {
                while (count == M_part.sparse().outerIndexPtr()[col + 1]) {
                    col += 1;
                }
                row = M_part.sparse().innerIndexPtr()[idx];
                sparsecolIndex[col].push_back(std::make_pair(row, M_part.sparse().valuePtr()[idx]));
                count += 1;
            }

        }
        else {
            M_skt_V = MatrixType(DsMatrix(full_m,d_v));
            M_skt_U = MatrixType(DsMatrix(part_n,d_u));
        }
        DsMatrix V_sketched_local = DsMatrix(d_v,parameter.k); //T'V
        DsMatrix U_sketched_local = DsMatrix(d_u,parameter.k); //S'U
        DsMatrix U_sketched_global_sum = DsMatrix(d_u, parameter.k);

        std::vector<int> index_sampled_m(d_u, 0);
        std::vector<int> index_sampled_n(d_v, 0);
        std::uniform_int_distribution<int> distribution_n(0, part_n - 1);
        std::uniform_int_distribution<int> distribution_m(0, full_m - 1);
        std::default_random_engine generator(SEED);

        double check_time = 0.0;

        double local_norm = M_part.squaredNorm();
        double total_norm;
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Reduce(&local_norm, &total_norm, 1, MPI_DOUBLE, MPI_SUM, ROOT_RANK, MPI_COMM_WORLD);

        for (int epoch = 0; epoch <= parameter.max_epoch; ++epoch) {
            timer.start();

            // average U
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Allreduce(U_local.data(), U_global_sum.data(), static_cast<int>(U_local.size()),
                          MPI_FLOAT_TYPE, MPI_SUM, MPI_COMM_WORLD);
            U_local = U_global_sum / world_size;

            //time_elapsed += timer_stop();
            timer.stop();
            total_time_elapsed+= timer.getElapsedTime();
            check_time = timer.getElapsedTime();

//            if (worker_rank == ROOT_RANK) {
//                std::cout << "Average U:" << check_time << std::endl;
//            }

            if (parameter.verbose_interval > 0 && epoch % parameter.verbose_interval == 0) {
                // compute loss
                double local_loss = compute_loss(M_part, U_local, V_part);
                double total_loss;
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_Reduce(&local_loss, &total_loss, 1, MPI_DOUBLE, MPI_SUM, ROOT_RANK, MPI_COMM_WORLD);

                if (worker_rank == ROOT_RANK) {
                    total_loss = total_loss / total_norm;
                    std::cout << "Iteration " << epoch << ": time=" << total_time_elapsed << ", loss=" << total_loss << std::endl;
                    fout<<total_time_elapsed<<'\t'<<total_loss<<endl;
                }
            }

            // timer_start();

            for (int sub_iter = 0; sub_iter <= parameter.max_sub_iter; ++sub_iter) {
                double mu = 0.01 * (parameter.alpha +  (epoch * parameter.max_sub_iter + sub_iter) * parameter.beta);

                timer.start();
                // subsampled inducies
                for (auto &idx : index_sampled_n)
                    idx = distribution_n(generator);

                // subsampling sketching V
                if (M_part.isSparse()) {
                    // M_part.sparse().makeCompressed();
                    M_skt_V.sparse().setZero();
                    std::vector<Triplet<FLOAT>> tripletList;
                    tripletList.reserve(M_part.sparse().nonZeros());
                    for (int i = 0; i < d_v; i++) {
                        const int idx = index_sampled_n[i];
                        V_sketched_local.row(i) = V_part.row(idx);
                        for (const auto &row_val_pair : sparsecolIndex[idx]) {
                            tripletList.push_back(Triplet<FLOAT>(static_cast<int>(row_val_pair.first), i, row_val_pair.second));
                        }
                    }
                    M_skt_V.sparse().setFromTriplets(tripletList.cbegin(), tripletList.cend());
                }
                else {
                    for (int i = 0; i < d_v; i++) {
                        const int idx = index_sampled_n[i];
                        V_sketched_local.row(i) = V_part.row(idx);
                        M_skt_V.dense().col(i) = M_part.dense().col(idx);
                    }

                }
                timer.stop();
                check_time = timer.getElapsedTime();
                total_time_elapsed+= timer.getElapsedTime();

                timer.start();
                // update U
                coordinate_descent(U_local, M_skt_V, V_sketched_local, mu);
                timer.stop();
                check_time = timer.getElapsedTime();
                total_time_elapsed+= timer.getElapsedTime();

                timer.start();
                // subsampled indices for U
                for (auto &idx : index_sampled_m)
                    idx = distribution_m(generator);

                // subsampling sketching U
                if (M_part.isSparse()) {
                    // M_part.sparse().makeCompressed();
                    M_skt_U.sparse().setZero();
                    std::vector<Triplet<FLOAT>> tripletList;
                    tripletList.reserve(M_part.sparse().nonZeros());
                    for (int i = 0; i < d_u; i++) {
                        const int idx = index_sampled_m[i];
                        U_sketched_local.row(i) = U_local.row(idx);
                        //M_sketched_part.col[i] = M_part.row(idx);
                        for (const auto &col_val_pair : sparserowIndex[idx]) {
                            tripletList.push_back(Triplet<FLOAT>(static_cast<int>(col_val_pair.first), i, col_val_pair.second));
                        }
                    }
                    M_skt_U.sparse().setFromTriplets(tripletList.cbegin(), tripletList.cend());
                }
                else {
                    for (int i = 0; i < d_u; i++) {
                        const int idx = index_sampled_m[i];
                        U_sketched_local.row(i) = U_local.row(idx);
                        M_skt_U.dense().col(i) = M_part.dense().row(idx);
                    }

                }
                timer.stop();
                check_time = timer.getElapsedTime();
                total_time_elapsed+= timer.getElapsedTime();

                timer.start();
                // average U_skt
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_Allreduce(U_sketched_local.data(), U_sketched_global_sum.data(), static_cast<int>(U_sketched_local.size()),
                              MPI_FLOAT_TYPE, MPI_SUM, MPI_COMM_WORLD);
                U_sketched_local = U_sketched_global_sum / world_size;

                timer.stop();
                check_time = timer.getElapsedTime();
                total_time_elapsed+= timer.getElapsedTime();

                timer.start();

                // update V
                coordinate_descent(V_part, M_skt_U, U_sketched_local, mu);
                timer.stop();
                check_time = timer.getElapsedTime();
                total_time_elapsed+= timer.getElapsedTime();

            }
        }

        if (worker_rank == 0) {
            cout << " total time elapsed for computation: " << total_time_elapsed << endl;
        }
    }



};
#endif //NMF_SYNCSKTUV_H
