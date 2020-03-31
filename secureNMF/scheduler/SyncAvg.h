#ifndef NMF_SYNCAVG_H
#define NMF_SYNCAVG_H

#include "SchedulerAbstract.h"
class SyncAvg : public SchedulerAbstract{
public:
    SyncAvg (const Parameter& parameter): SchedulerAbstract(parameter){

    }

    void calculate_NMF()override{
        ofstream fout;
//        cout<<worker_rank<<endl;
        if(worker_rank==ROOT_RANK){
            fout.open((parameter.output_path+"_time-loss.txt").c_str());
//            cout<<(parameter.output_path+"_time-loss.txt")<<endl;
        }
        Monitor timer;
        double time_elapsed = 0;
        double total_time_elapsed=0;
        double local_loss=0;
        double total_loss=0;

        int part_n = static_cast<int>(M_part.cols());

        DsMatrix U_global_sum = DsMatrix(full_m, parameter.k);

        double local_norm = M_part.squaredNorm();
        double total_norm;
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Reduce(&local_norm, &total_norm, 1, MPI_DOUBLE, MPI_SUM, ROOT_RANK, MPI_COMM_WORLD);

        for (int epoch = 0; epoch <= parameter.max_epoch; ++epoch) {
            time_elapsed = 0;
            timer.start();

//            if(worker_rank==ROOT_RANK){
//                //server
//
//            }
//            else{
//                //slaves
//
//            }
            // average U
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Allreduce(U_local.data(), U_global_sum.data(), static_cast<int>(U_local.size()),
                          MPI_FLOAT_TYPE, MPI_SUM, MPI_COMM_WORLD);
            U_local = U_global_sum / world_size;

            timer.stop();
            time_elapsed += timer.getElapsedTime();

            if (parameter.verbose_interval > 0 && epoch % parameter.verbose_interval == 0) {
                local_loss = compute_loss(M_part, U_local, V_part);
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_Reduce(&local_loss, &total_loss, 1, MPI_DOUBLE, MPI_SUM, ROOT_RANK, MPI_COMM_WORLD);
            }

            timer.start();
            for (int sub_iter = 0; sub_iter <= parameter.max_sub_iter; ++sub_iter) {
                double mu = parameter.alpha + (epoch * parameter.max_sub_iter + sub_iter) * parameter.beta;

                coordinate_descent(U_local, M_part, V_part, mu);
                coordinate_descent(V_part, M_part.transpose(), U_local, mu);
            }
            timer.stop();
            time_elapsed += timer.getElapsedTime();
            total_time_elapsed+= time_elapsed;

            if (parameter.verbose_interval > 0 && epoch % parameter.verbose_interval == 0) {
                if (worker_rank == ROOT_RANK) {
                    total_loss = total_loss / total_norm;
                    std::cout << "Iteration " << epoch << ": time=" << total_time_elapsed << ", loss=" << total_loss << std::endl;
                    fout<<total_time_elapsed<<'\t'<<total_loss<<endl;
                }
            }


        }

        if (worker_rank == 0) {
            cout << " total time elapsed for computation: " << total_time_elapsed << endl;
        }
        fout.close();
    }

};


#endif //NMF_SYNCAVG_H
