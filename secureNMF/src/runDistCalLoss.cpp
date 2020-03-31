#include "../util/base.h"
#include "../util/parameter.h"
#include "../util/monitor.h"
#include "../scheduler/SyncAvg.h"
#include "../scheduler/SyncSktU.h"
#include "../scheduler/AsyncAvgInd.h"
#include "../scheduler/AsyncAvgDump.h"
#include "../scheduler/AsyncSktV.h"
#include "../util/serialization.h"


//TODO: change to MPI environment
int main(int argc, char *argv[]) {
    int worker_rank;
    int world_size;
    MPI_Init(NULL, NULL);
    MPI_Comm comm_world = MPI_COMM_WORLD;
    MPI_Comm_rank(MPI_COMM_WORLD, &worker_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    Parameter parameter;

    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "produce help message")
//            ("k", po::value<int>(&(parameter.k))->default_value(100))
//            ("solution_type", po::value<string>(&(parameter.solution_type))->default_value("AsyncAvg2"))
//            ("max_epoch", po::value<int>(&(parameter.max_epoch))->default_value(100))
//            ("max_sub_iter", po::value<int>(&(parameter.max_sub_iter))->default_value(10))
//            ("row_ratio", po::value<double>(&(parameter.row_ratio))->default_value(0.1))
//            ("col_ratio", po::value<double>(&(parameter.col_ratio))->default_value(0.1))
//            ("alpha", po::value<double>(&(parameter.alpha))->default_value(100))
//            ("beta", po::value<double>(&(parameter.beta))->default_value(10))
//            ("verbose_interval", po::value<int>(&(parameter.verbose_interval))->default_value(1))
//            ("dump_interval", po::value<double>(&(parameter.dump_interval))->default_value(0.5))
//            ("transpose_file_matrix", po::value<bool>(&(parameter.transpose_file_matrix))->default_value(false))
            ("input_file", po::value<string>(&(parameter.input_file))->default_value(
                    "../experiment/face/"),
             "path to graph file")
            ("o_path", po::value<string>(&(parameter.output_path))->default_value(
                    "../experiment/face/"), "path to output prefix")
            ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << endl;
        return 0;
    }

    ofstream fout;
    if(worker_rank==0)
        fout.open((parameter.output_path+"_time-loss.txt").c_str());

    string dump_file = parameter.input_file+"_"+std::to_string(worker_rank)+".dump";
    ifstream fin(dump_file.c_str());
    if(!fin.good()){cerr<<"error in openning file "<<dump_file<<endl; exit(-1);}
    boost::archive::binary_iarchive ia(fin);

    double total_norm=0;
    double local_norm=0;
    if(worker_rank)
        ia>>local_norm;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&local_norm, &total_norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    //read local loss for each dump iteration
    for(int dump = 0; ; ++dump){
        try {
            double total_loss=0;
            double total_time=0;
            double local_time=0;
            double local_loss=0;
//            for(int i=1; i<num_peers;++i) {
//                int slave_id = i - 1;
//                double local_loss;
//                *ias[i]>>local_time>>local_loss;
//                total_loss+=local_loss;
//                time_elapsed+=local_time;
//            }

            if(worker_rank){
                ia>>local_time>>local_loss;
            }
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Reduce(&local_loss, &total_loss, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Reduce(&local_time, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            if(worker_rank==0) {
                total_loss /= total_norm;
                total_time /= (world_size - 1);
                cout << "dump " << dump << ", time elapsed: " << total_time << ", total loss: " << total_loss << endl;
                fout << total_time << '\t' << total_loss << endl;
            }
        }
        catch (const boost::archive::archive_exception::exception_code& e){
            cout<<"exception code "<<e<<" detected"<<endl;
            break;
        }
        catch (...){
            cerr<<"Other exeptions\n";
            break;
        }
    }



    return 0;
}
