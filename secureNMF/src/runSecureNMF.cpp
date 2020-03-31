//
// Created by danhao on 1/11/19.
//

#include "../util/base.h"
#include "../util/parameter.h"
#include "../util/monitor.h"
#include "../scheduler/SyncAvg.h"
#include "../scheduler/SyncSktU.h"
#include "../scheduler/SyncSktV.h"
#include "../scheduler/SyncSktUV.h"
#include "../scheduler/AsyncAvgInd.h"
#include "../scheduler/AsyncAvgDump.h"
#include "../scheduler/AsyncAvg.h"
#include "../scheduler/AsyncSktV.h"


int main(int argc, char *argv[]) {

    Parameter parameter;

    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "produce help message")
            ("k", po::value<int>(&(parameter.k))->default_value(100))
            ("solution_type", po::value<string>(&(parameter.solution_type))->default_value("SyncAvg"))
            ("max_epoch", po::value<int>(&(parameter.max_epoch))->default_value(100))
            ("max_sub_iter", po::value<int>(&(parameter.max_sub_iter))->default_value(10))
            ("row_ratio", po::value<double>(&(parameter.row_ratio))->default_value(0.1))
            ("col_ratio", po::value<double>(&(parameter.col_ratio))->default_value(0.1))
            ("alpha", po::value<double>(&(parameter.alpha))->default_value(1000))
            ("beta", po::value<double>(&(parameter.beta))->default_value(100))
            ("verbose_interval", po::value<int>(&(parameter.verbose_interval))->default_value(1))
            ("balanced", po::value<bool>(&(parameter.balanced))->default_value(true))
            ("dump_interval", po::value<double>(&(parameter.dump_interval))->default_value(0.5))
            ("transpose_file_matrix", po::value<bool>(&(parameter.transpose_file_matrix))->default_value(false))
            ("input_file", po::value<string>(&(parameter.input_file))->default_value(
                    "../data/dataset/face.m"),
             "path to graph file")
            ("sor", po::value<double>(&(parameter.sor))->default_value(1))
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

    SchedulerAbstract *worker;
    if (parameter.solution_type == "SyncAvg") {
        worker = new SyncAvg(parameter);
    } else if (parameter.solution_type == "AsyncAvg") {
        worker = new AsyncAvg(parameter);
    } else if (parameter.solution_type == "AsyncSktV") {
        worker = new AsyncSktV(parameter);
    } else if (parameter.solution_type == "SyncSktU") {
        worker = new SyncSktU(parameter);
    } else if (parameter.solution_type == "SyncSktV") {
        worker = new SyncSktV(parameter);
    }else if (parameter.solution_type == "SyncSktUV") {
        worker = new SyncSkt_UV(parameter);
    }else {
        cerr << "unrecognized synchronization mode\n";
        exit(-1);
    }

    std::ifstream fin(parameter.input_file);
    if (fin.fail()) {
        cerr << "unable to load file" << parameter.input_file << endl;
        exit(1);
    }

    worker->calculate_NMF();

    //cout<<"computation finished\n";
    delete worker;
    //cout<<"worker deleted!\n";
    return 0;
}
