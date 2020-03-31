#include "../util/base.h"
#include "../util/parameter.h"
#include "../util/monitor.h"
#include "../scheduler/SyncAvg.h"
#include "../scheduler/SyncSktU.h"
#include "../scheduler/AsyncAvgInd.h"
#include "../scheduler/AsyncAvgDump.h"
#include "../scheduler/AsyncSktV.h"
#include "../util/serialization.h"

double compute_loss(const MatrixType &M, const DsMatrix& U, const DsMatrix& V)
{
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

    assert(num_peers>1);

    vector<std::unique_ptr<boost::archive::binary_iarchive>> ias(num_peers);
    vector<std::ifstream> fins(num_peers);

    for(int i=0; i<num_peers;++i){
        string dump_file = parameter.output_path+"_"+std::to_string(i)+".dump";
        fins[i].open(dump_file.c_str());
        assert(fins[i].good());
        ias[i]=std::make_unique<boost::archive::binary_iarchive>(fins[i]);
    }


    vector<DsMatrix> M_locals(num_peers-1);
    vector<DsMatrix> V_locals(num_peers-1);
    double total_norm=0;

    //read M_locals
    for(int i=1; i<num_peers;++i){
        int slave_id = i-1;
        *ias[i]>>M_locals[slave_id];
        total_norm += M_locals[slave_id].squaredNorm();
    }

    //read U_global and V_local for each dump iteration
    for(int dump = 0; ; ++dump){
        try {
            double total_loss=0;

            double time_elapsed;
            DsMatrix U_global;
            *ias[0]>>time_elapsed;
            *ias[0]>>U_global;

            for(int i=1; i<num_peers;++i){
                int slave_id = i-1;
                *ias[i]>>V_locals[slave_id];
                total_loss += compute_loss(M_locals[slave_id], U_global, V_locals[slave_id]);
            }

            total_loss /= total_norm;

            cout<<"dump "<<dump<<", time elapsed: "<<time_elapsed<<", total loss: "<<total_loss<<endl;

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