#ifndef MATRIX_READER_HPP_
#define MATRIX_READER_HPP_
#include "basics.hpp"

#include <iostream>
#include <vector>

namespace NPDivs {

std::vector< std::vector<double> > matrix_from_csv(
        std::istream &in, size_t dim = 0);

std::vector< std::vector< std::vector<double> > >
matrices_from_csv(std::istream &in, size_t dim = 0);

} // end namespace

#endif
