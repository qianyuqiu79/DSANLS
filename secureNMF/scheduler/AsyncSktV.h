#ifndef NMF_ASYNCSKT_H
#define NMF_ASYNCSKT_H

#include <future>
#include <unordered_set>
#include "../util/serialization.h"
#include "SchedulerAbstract.h"

class AsyncSktV : public SchedulerAbstract {
public:
    AsyncSktV(const Parameter &parameter) : SchedulerAbstract(parameter, noprapere_tag()) {
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
//        dump_file = parameter.output_path + "_" + std::to_string(worker_rank) + ".dump";
//        dump_out.open(dump_file.c_str());
//        if (!dump_out.good()) {
//            cerr << "fail to open dump file\n";
//            exit(-1);
//        }
        local_loss_store.resize(parameter.max_epoch+1);
        local_time_store.resize(parameter.max_epoch+1);
    }

    void initialize_data() override {
        // worker_rank=0 -> server; init=slave 0, but no work to do
        // slaves
        int slaves = world_size - 1;
        int my_slave_id = worker_rank - 1;
        if (worker_rank == 0) {
            my_slave_id = 0;
        }
        n_partition = vector<int>(slaves + 1, 0);

        if (!parameter.input_file.empty()) {
            M_part.read_matrix(parameter.input_file.c_str(), my_slave_id, slaves, n_partition,
                               parameter.transpose_file_matrix);
        } else {
            // M_part.dense_random_matrix(100, 1000, rank, size, n_partition);
            M_part.sparse_test_matrix(1000, 1000, my_slave_id, slaves, n_partition);
            //std::cout << "Need to provide filename of matrix." << std::endl;
            //return -1;
        }

        full_m = M_part.rows();
        full_n = n_partition[slaves];
        part_n = M_part.cols();

        U_local = DsMatrix(DsMatrix::Random(full_m, parameter.k).array() + 1.0) * (parameter.rand_upper_bound / 2);
        V_part = DsMatrix(DsMatrix::Random(part_n, parameter.k).array() + 1.0) * (parameter.rand_upper_bound / 2);

        m_partition = std::vector<int>(slaves + 1, 0);
        split_idx(full_m, slaves, m_partition, parameter.balanced);
        part_m = m_partition[my_slave_id + 1] - m_partition[my_slave_id];
        U_part = DsMatrix(U_local.block(m_partition[my_slave_id], 0, part_m, parameter.k));
        if(worker_rank==0)U_local*=0;

        local_norm = M_part.squaredNorm();
        if(worker_rank==0){local_norm = 0;}
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Reduce(&local_norm, &total_norm, 1, MPI_DOUBLE, MPI_SUM, ROOT_RANK, MPI_COMM_WORLD);
//        cout<<U_local.size()<<" "<<V_part.size()<<endl;
//        cout<<U_local.squaredNorm()<<endl;
    }

    void calculate_NMF() {
//        boost::archive::binary_oarchive oa(dump_out);

        double check_time = 0.0;

        DsMatrix U_global_sum = DsMatrix(full_m, parameter.k);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(U_local.data(), U_global_sum.data(), static_cast<int>(U_local.size()),
                      MPI_FLOAT_TYPE, MPI_SUM, MPI_COMM_WORLD);
        U_local = U_global_sum / (world_size-1);

        if (worker_rank == ROOT_RANK) {
            //server
            Monitor timer;
            double time_elapsed;

            DsMatrix U_global(U_local);
            DsMatrix recv_buffer = U_global;

            int probed = 0;
            MPI_Status status;
//            vector<MPI_Request> sending_queue();
            std::unordered_set<int> pending_updates;

            int output_counter = 0;
            int total_finished = 0;

            timer.start();
            int update_round=1;
            for (int server_iteration = 0;; ++server_iteration) {
                time_elapsed = timer.getElapsedTime();
//                cout << "server iteration: " << server_iteration << endl;
                pending_updates.clear();
                do {
                    probed = 0;
                    MPI_Message msg;
                    MPI_Improbe(MPI_ANY_SOURCE, UPDATE_TAG, MPI_COMM_WORLD, &probed, &msg, &status);
                    int slave = status.MPI_SOURCE;
                    if (probed) {
                        ++update_round;
                        MPI_Mrecv(recv_buffer.data(), static_cast<int>(U_global.size()), MPI_FLOAT_TYPE,
                                  &msg, &status);

                        //update U_global
                        double omega =  (parameter.sor/(parameter.sor+update_round));
                        U_global = U_global * (1.0-omega) +
                                   omega * recv_buffer;
//                        U_global = U_global * (0.5) +
//                                   (0.5) * recv_buffer;
                        pending_updates.insert(slave);
                    }

                } while (probed);

                //send back U after all updates
                for (auto &slave:pending_updates) {
                    MPI_Send(U_global.data(), static_cast<int>(U_global.size()), MPI_FLOAT_TYPE,
                             slave,
                             UPDATE_TAG, MPI_COMM_WORLD);
                }

                int termination = 1;
                //termination order
                do {
                    probed = 0;
                    MPI_Message msg;
                    MPI_Improbe(MPI_ANY_SOURCE, TERMINATION_TAG, MPI_COMM_WORLD, &probed, &msg, &status);
                    if (probed) {
                        ++total_finished;
                        int dummy;
                        MPI_Status s;
                        MPI_Mrecv(&dummy, 1, MPI_INT, &msg, &s);
                    }
                } while (probed);

                if (total_finished >= world_size-1) {
                    //broadcast termination signal
                    int dummy;
                    for (int slave = 1; slave < world_size; ++slave) {
                        MPI_Send(&dummy, 1, MPI_INT, slave,
                                 TERMINATION_TAG, MPI_COMM_WORLD);
                    }
                    break;
                }

            }
        } else {
            //slaves
            int slaves = world_size - 1;
            int my_slave_id = worker_rank - 1;
//        DsMatrix U_recv_buffer = DsMatrix(full_m, parameter.k);

//            double local_norm;
            double total_time_elapsed = 0;
//            DsMatrix M_dump;
//            if (M_part.isSparse()) {
//                M_dump = DsMatrix(M_part.sparse());
//            } else {
//                M_dump = DsMatrix(M_part.dense());
//            }
//            local_norm=M_dump.squaredNorm();

//            oa<<local_norm;

            Monitor timer;
            cout<<setprecision(10);

            int d = int(part_n * parameter.col_ratio);

            MatrixType M_sketched_part;// = DsMatrix(full_m,d);

            std::vector<std::vector<std::pair<Index, FLOAT>>> sparsecolIndex;

            if (M_part.isSparse()) {
                M_sketched_part = MatrixType(SpMatrix(full_m, d));
                sparsecolIndex = std::vector<std::vector<std::pair<Index, FLOAT>>>(M_part.cols(), std::vector<std::pair<Index, FLOAT>>());
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
                M_sketched_part = MatrixType(DsMatrix(full_m,d)); // MT
            }
            DsMatrix V_sketched_local = DsMatrix(d,parameter.k); //T'V

            std::vector<int> index_sampled(d, 0);
            std::uniform_int_distribution<int> distribution(0, part_n - 1);
            std::default_random_engine generator(SEED);

            for (int epoch = 0; epoch <= parameter.max_epoch; ++epoch) {
                int termination;
                int probed = 0;
                MPI_Status status;
                if (!(epoch%parameter.verbose_interval)) {
                    double local_loss = compute_loss(M_part, U_local, V_part);
//                    cout << epoch <<","<<total_time_elapsed<<",";
//                    cout << local_loss <<","<<local_norm <<","<< local_loss/local_norm<<endl;

                    local_loss_store[epoch]=local_loss;
                    local_time_store[epoch]=total_time_elapsed;
//                    oa <<total_time_elapsed<<local_loss;
                }

                timer.start();

                //computation for local U and V
                for (int sub_iter = 0; sub_iter <= parameter.max_sub_iter; ++sub_iter) {
                    double mu = 0.01 * (parameter.alpha +  (epoch * parameter.max_sub_iter + sub_iter) * parameter.beta);

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
                    // update U
                    coordinate_descent(U_local, M_sketched_part, V_sketched_local, mu);

                    // update V
                    coordinate_descent(V_part, M_part.transpose(), U_local, mu);
                }

                //communicate with the server
                MPI_Send(U_local.data(), static_cast<int>(U_local.size()), MPI_FLOAT_TYPE, ROOT_RANK,
                         UPDATE_TAG, MPI_COMM_WORLD);

                MPI_Recv(U_local.data(), static_cast<int>(U_local.size()), MPI_FLOAT_TYPE, ROOT_RANK,
                         UPDATE_TAG, MPI_COMM_WORLD, &status);
                timer.stop();
                total_time_elapsed += timer.getElapsedTime();
            }

            int termination = 1;
            MPI_Status status;
            MPI_Send(&termination, 1, MPI_INT, ROOT_RANK,
                     TERMINATION_TAG, MPI_COMM_WORLD);
//                MPI_Barrier(MPI_COMM_WORLD);
            //wait server for termination order
            MPI_Recv(&termination, 1, MPI_INT, ROOT_RANK,
                     TERMINATION_TAG, MPI_COMM_WORLD, &status);
        }

        report_loss();
    }

};

#endif //NMF_ASYNCSKT_H
