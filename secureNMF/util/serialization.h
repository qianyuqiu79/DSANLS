#ifndef NMF_SERIALIZATION_H
#define NMF_SERIALIZATION_H

#include "base.h"
#include "../../common.h"
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

using namespace std;

//TODO: test
namespace boost{
    namespace serialization{

        template<   class Archive,
                class S,
                int Rows_,
                int Cols_,
                int Ops_,
                int MaxRows_,
                int MaxCols_>
        inline void serialize(Archive & ar,
                              Eigen::Matrix<S, Rows_, Cols_, Ops_, MaxRows_, MaxCols_> & matrix,
                              const unsigned int version)
        {
            int rows = matrix.rows();
            int cols = matrix.cols();
            ar & make_nvp("rows", rows);
            ar & make_nvp("cols", cols);
            matrix.resize(rows, cols); // no-op if size does not change!

            // always save/load row-major
            for(int r = 0; r < rows; ++r)
                for(int c = 0; c < cols; ++c)
                    ar & make_nvp("val", matrix(r,c));
        }
    } // namespace serialization
} // namespace boost

#endif //NMF_SERIALIZATION_H
