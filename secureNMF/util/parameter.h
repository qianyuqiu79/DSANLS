#ifndef NMF_PARAMETER_H
#define NMF_PARAMETER_H

#include "base.h"

using namespace std;

class Parameter {

public:

    string output_path = "****"; // set your own output path
    string input_file = "****"; // set your own input path

    const double rand_upper_bound = 1e-2;
    int k = 100;
    string solution_type = "SyncAvg";
    int max_epoch = 100;
    int max_sub_iter = 10;
    double row_ratio = 0.1, col_ratio = 0.1;
    double alpha = 100.0, beta = 10.0;
    int verbose_interval = 1;
    bool transpose_file_matrix = false;
    double dump_interval = 1;
    bool balanced=true;
    double sor = 1;

    string get_all() const {

        std::stringstream ss;
        ss << "k: " << k << endl;
        ss << "solution_type: " << solution_type << endl;
        ss << "max_epoch: " << max_epoch << endl;
        ss << "max_sub_iter: " << max_sub_iter << endl;
        ss << "row_ratio: " << row_ratio << endl;
        ss << "col_ratio: " << col_ratio << endl;
        ss << "alpha: " << alpha << endl;
        ss << "beta: " << beta << endl;
        ss << "verbose_interval: " << verbose_interval << endl;
        ss << "dump interval time: " << dump_interval << endl;
        ss << "transpose_file_matrix: " << transpose_file_matrix << endl;
        ss << "output_path: " << output_path << endl;
        ss << "input_file: " << input_file << endl;
        ss << "balanced: " << balanced <<endl;
        ss << "sor: " <<sor<<endl;
        return ss.str();
    }
};

#endif //NMF_PARAMETER_H
