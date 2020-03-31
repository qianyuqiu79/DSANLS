#ifndef NMF_BASE_H
#define NMF_BASE_H

#include "mpi.h"
#include <string>
#include <iostream>
#include <sstream>
#include <random>
#include <fstream>
#include <memory>
#include <iomanip>
#include <thread>

#define ROOT_RANK 0
#define SEED 0

typedef float value_type;

const value_type min_not_zero_value = 1e-7;

using std::string;


//BOOST
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>

namespace po = boost::program_options;

#endif //NMF_BASE_H
