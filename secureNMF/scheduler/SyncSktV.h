#ifndef NMF_SYNCSKT_SV_H
#define NMF_SYNCSKT_SV_H

#include <unordered_map>
#include "SchedulerAbstract.h"

class SyncSktV : public SchedulerAbstract{
public:
    SyncSktV(const Parameter& parameter): SchedulerAbstract(parameter){

    }

    void calculate_NMF(){
        ofstream fout;
//        cout<<worker_rank<<endl;
        if(worker_rank==ROOT_RANK){
            fout.open((parameter.output_path+"_time-loss.txt").c_str());
//            cout<<(parameter.output_path+"_time-loss.txt")<<endl;
        }
        Monitor timer;
        double total_time_elapsed=0;
        int d = int(part_n * parameter.col_ratio);

        DsMatrix U_global_sum = DsMatrix(full_m, parameter.k);

        MatrixType M_sketched_part;// = DsMatrix(full_m,d);

        std::vector<std::vector<std::pair<Index, FLOAT>>> sparsecolIndex;
//        std::unordered_map<int, std::unordered_map<int, std::pair<Index, FLOAT>>> sparsecolIndex;

        if (M_part.isSparse()) {
            M_sketched_part = MatrixType(SpMatrix(full_m, d));
            sparsecolIndex = std::vector<std::vector<std::pair<Index, FLOAT>>>(M_part.cols(), std::vector<std::pair<Index, FLOAT>>(0));

// M_part.sparse().makeCompressed();

            int col = 0;
            int row;
            int count = 0;
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
            M_sketched_part = MatrixType(DsMatrix(full_m,d));
        }
        DsMatrix V_sketched_local = DsMatrix(d,parameter.k); //T'V
        std::vector<int> index_sampled(d, 0);
        std::uniform_int_distribution<int> distribution(0, part_n - 1);
        std::default_random_engine generator(SEED);

//        double check_time = 0.0;

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
//            check_time = timer.getElapsedTime();

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
                // update U
                coordinate_descent(U_local, M_sketched_part, V_sketched_local, mu);
                timer.stop();
//                check_time = timer.getElapsedTime();
                total_time_elapsed+= timer.getElapsedTime();
//                if (worker_rank == ROOT_RANK) {
//                    std::cout << "U: column=" << U_local.cols() << " row=" << U_local.rows()
//                              << "\tM: column=" << M_part.cols() << " row=" << M_part.rows()
//                              << "\tV: column=" << V_part.cols() << " row=" << V_part.rows()
//                              << std::endl;
//                    std::cout << "Update U:" << check_time << std::endl;
//                }

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
                        V_sketched_local.row(i) = V_part.row(idx);
                        for (const auto &row_val_pair : sparsecolIndex[idx]) {
                            tripletList.push_back(Triplet<FLOAT>(static_cast<int>(row_val_pair.first), i, row_val_pair.second));
                        }
                    }
                    M_sketched_part.sparse().setFromTriplets(tripletList.cbegin(), tripletList.cend());
                }
                else {
                    for (int i = 0; i < d; i++) {
                        const int idx = index_sampled[i];
                        V_sketched_local.row(i) = V_part.row(idx);
                        M_sketched_part.dense().col(i) = M_part.dense().col(idx);
                    }

                }
                timer.stop();
//                check_time = timer.getElapsedTime();
                total_time_elapsed+= timer.getElapsedTime();
//                if (worker_rank == ROOT_RANK) {
//                    std::cout << "Subsample:" << check_time << std::endl;
//                }

                timer.start();

                MPI_Barrier(MPI_COMM_WORLD);
                MPI_Allreduce(U_local.data(), U_global_sum.data(), static_cast<int>(U_local.size()),
                              MPI_FLOAT_TYPE, MPI_SUM, MPI_COMM_WORLD);
                U_local = U_global_sum / world_size;
                timer.stop();
//                check_time = timer.getElapsedTime();
                total_time_elapsed+= timer.getElapsedTime();

                timer.start();

                // update V
                coordinate_descent(V_part, M_part.transpose(), U_local, mu);
                timer.stop();
//                check_time = timer.getElapsedTime();
                total_time_elapsed+= timer.getElapsedTime();
//                if (worker_rank == ROOT_RANK) {
//                    std::cout << "V: column=" << V_part.cols() << " row=" << V_part.rows()
//                              << "\tM: column=" << M_sketched_part.cols() << " row=" << M_sketched_part.rows()
//                              << "\tU: column=" << V_sketched_local.cols() << " row=" << V_sketched_local.rows()
//                              << std::endl;
//                    std::cout << "Update V:" << check_time << std::endl;
//                }


                // std::cout << U_local.maxCoeff() << " " << V_part.maxCoeff() << std::endl;

            }

            // time_elapsed += timer_stop();
        }

        if (worker_rank == 0) {
            cout << " total time elapsed for computation: " << total_time_elapsed << endl;
        }
    }



};
#endif //NMF_SYNCSKT_SV_H
