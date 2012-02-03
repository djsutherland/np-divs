#ifndef MATRIX_READER_HPP_
#define MATRIX_READER_HPP_
#include "basics.hpp"

#include <iostream>
#include <vector>

#include <flann/util/matrix.h>

namespace NPDivs {

std::vector< std::vector<double> > matrix_vector_from_csv(
        std::istream &in, size_t dim = 0);

std::vector< std::vector< std::vector<double> > >
matrices_vector_from_csv(std::istream &in, size_t dim = 0);

flann::Matrix<double> matrix_from_csv(std::istream &in);
flann::Matrix<double>* matrices_from_csv(std::istream &in, size_t &n);

template <typename T>
void matrix_to_csv(std::ostream &out, flann::Matrix<T> mat);

template <typename T>
void matrix_array_to_csv(std::ostream &out, flann::Matrix<T>* mat, size_t n);


////////////////////////////////////////////////////////////////////////////////
// implementations of templates
template <typename T>
void matrix_to_csv(std::ostream &out, flann::Matrix<T> mat) {
    for (size_t i = 0; i < mat.rows; i++) {
        out << mat[i][0];
        for (size_t j = 1; j < mat.cols; j++)
            out << ", " << mat[i][j];
        out << "\n";
    }
}

template <typename T>
void matrix_array_to_csv(std::ostream &out, flann::Matrix<T>* mats, size_t n) {
    for (size_t i = 0; i < n; i++) {
        matrix_to_csv(out, mats[i]);
        out << "\n";
    }
    out << "\n";
}


} // end namespace
#endif
