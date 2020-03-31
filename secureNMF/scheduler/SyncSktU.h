#ifndef NMF_SYNCSKT_H
#define NMF_SYNCSKT_H

#include "SchedulerAbstract.h"

class SyncSktU : public SchedulerAbstract{
public:
    SyncSktU(const Parameter& parameter): SchedulerAbstract(parameter){

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
        int d = int(full_m * parameter.row_ratio);
        Index part_n = M_part.cols();

        DsMatrix U_global_sum = DsMatrix(full_m, parameter.k);

        MatrixType M_sketched_part;// = DsMatrix(part_n, d);

        std::vector<std::vector<std::pair<Index, FLOAT>>> sparserowIndex;

        if (M_part.isSparse()) {
            sparserowIndex = std::vector<std::vector<std::pair<Index, FLOAT>>>(M_part.rows(), std::vector<std::pair<Index, FLOAT>>());

            M_sketched_part = MatrixType(SpMatrix(part_n, d));
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
        }
        else {
            M_sketched_part = MatrixType(DsMatrix(part_n, d));
        }
        DsMatrix U_sketched_local = DsMatrix(d, parameter.k);
        DsMatrix U_sketched_global_sum = DsMatrix(d, parameter.k);

        std::vector<int> index_sampled(d, 0);
        std::uniform_int_distribution<int> distribution(0, full_m - 1);
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
                // update U
                coordinate_descent(U_local, M_part, V_part, mu);
                timer.stop();
                check_time = timer.getElapsedTime();
                total_time_elapsed+= timer.getElapsedTime();

                timer.start();

                // subsampled inducies
                for (auto &idx : index_sampled)
                    idx = distribution(generator);

                // subsampling sketching
                if (M_part.isSparse()) {
                    // M_part.sparse().makeCompressed();
                    M_sketched_part.sparse().setZero();
                    std::vector<Triplet<FLOAT>> tripletList;
                    tripletList.reserve(M_part.sparse().nonZeros());
                    for (int i = 0; i < d; i++) {
                        const int idx = index_sampled[i];
                        U_sketched_local.row(i) = U_local.row(idx);
                        //M_sketched_part.col[i] = M_part.row(idx);
                        for (const auto &col_val_pair : sparserowIndex[idx]) {
                            tripletList.push_back(Triplet<FLOAT>(static_cast<int>(col_val_pair.first), i, col_val_pair.second));
                        }
                    }
                    M_sketched_part.sparse().setFromTriplets(tripletList.cbegin(), tripletList.cend());
                }
                else {
                    for (int i = 0; i < d; i++) {
                        const int idx = index_sampled[i];
                        U_sketched_local.row(i) = U_local.row(idx);
                        M_sketched_part.dense().col(i) = M_part.dense().row(idx);
                    }

                }
                timer.stop();
                check_time = timer.getElapsedTime();
                total_time_elapsed+= timer.getElapsedTime();
                timer.start();

                MPI_Barrier(MPI_COMM_WORLD);
                MPI_Allreduce(U_sketched_local.data(), U_sketched_global_sum.data(), static_cast<int>(U_sketched_local.size()),
                              MPI_FLOAT_TYPE, MPI_SUM, MPI_COMM_WORLD);
                U_sketched_local = U_sketched_global_sum / world_size;
                timer.stop();
                check_time = timer.getElapsedTime();
                total_time_elapsed+= timer.getElapsedTime();

                timer.start();

                // update V
                coordinate_descent(V_part, M_sketched_part, U_sketched_local, mu);
                timer.stop();
                check_time = timer.getElapsedTime();
                total_time_elapsed+= timer.getElapsedTime();
                // std::cout << U_local.maxCoeff() << " " << V_part.maxCoeff() << std::endl;

            }

            // time_elapsed += timer_stop();
        }

        if (worker_rank == 0) {
            cout << " total time elapsed for computation: " << total_time_elapsed << endl;
        }
    }



};

#endif //NMF_SYNCSKT_H
