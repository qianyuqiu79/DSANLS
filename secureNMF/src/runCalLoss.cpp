#include "../util/base.h"
#include "../util/parameter.h"
#include "../util/monitor.h"
#include "../scheduler/SyncAvg.h"
#include "../scheduler/SyncSktU.h"
#include "../scheduler/AsyncAvgInd.h"
#include "../scheduler/AsyncAvgDump.h"
#include "../scheduler/AsyncSktV.h"
#include "../util/serialization.h"

int main(int argc, char *argv[]) {

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
//            ("input_file", po::value<string>(&(parameter.input_file))->default_value(
//                    "../data/dataset/face.m"),
//             "path to graph file")
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

    int num_peers;

    std::cout<<"Enter num of peers: ";
    std::cin>>num_peers;
    std::cout<<endl;

    ofstream fout((parameter.output_path+"_time-loss.txt").c_str());

    assert(num_peers>1);

    vector<std::unique_ptr<boost::archive::binary_iarchive>> ias(num_peers);
    vector<std::ifstream> fins(num_peers);

    double total_norm=0;
    for(int i=0; i<num_peers;++i){
        string dump_file = parameter.output_path+"_"+std::to_string(i)+".dump";
        fins[i].open(dump_file.c_str());
        assert(fins[i].good());
        ias[i]=std::make_unique<boost::archive::binary_iarchive>(fins[i]);
    }

    for(int i=1; i<num_peers;++i){
        int slave_id = i-1;
        double local_norm=0;
        *ias[i]>>local_norm;
        total_norm += local_norm;
    }

    //read local loss for each dump iteration
    for(int dump = 0; ; ++dump){
        try {
            double total_loss=0;
            double time_elapsed=0;
            double local_time=0;

            for(int i=1; i<num_peers;++i) {
                int slave_id = i - 1;
                double local_loss;
                *ias[i]>>local_time>>local_loss;
                total_loss+=local_loss;
                time_elapsed+=local_time;
            }
            total_loss /= total_norm;
            time_elapsed /= (num_peers-1);
            cout<<"dump "<<dump<<", time elapsed: "<<time_elapsed<<", total loss: "<<total_loss<<endl;
            fout<<time_elapsed<<'\t'<<total_loss<<endl;
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