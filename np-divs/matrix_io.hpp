/*******************************************************************************
 * Copyright (c) 2012, Dougal J. Sutherland (dsutherl@cs.cmu.edu).             *
 * All rights reserved.                                                        *
 *                                                                             *
 * Redistribution and use in source and binary forms, with or without          *
 * modification, are permitted provided that the following conditions are met: *
 *                                                                             *
 *     * Redistributions of source code must retain the above copyright        *
 *       notice, this list of conditions and the following disclaimer.         *
 *                                                                             *
 *     * Redistributions in binary form must reproduce the above copyright     *
 *       notice, this list of conditions and the following disclaimer in the   *
 *       documentation and/or other materials provided with the distribution.  *
 *                                                                             *
 *     * Neither the name of Carnegie Mellon University nor the                *
 *       names of the contributors may be used to endorse or promote products  *
 *       derived from this software without specific prior written permission. *
 *                                                                             *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" *
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE   *
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE  *
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE   *
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR         *
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF        *
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS    *
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN     *
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)     *
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE  *
 * POSSIBILITY OF SUCH DAMAGE.                                                 *
 ******************************************************************************/
#ifndef NPDIVS_MATRIX_READER_HPP_
#define NPDIVS_MATRIX_READER_HPP_
#include "basics.hpp"

#include <iostream>
#include <string>
#include <vector>

#include <flann/util/matrix.h>

namespace npdivs {

std::vector< std::vector<double> > matrix_vector_from_csv(
        std::istream &in, size_t dim = 0);

std::vector< std::vector< std::vector<double> > >
matrices_vector_from_csv(std::istream &in, size_t dim = 0);

std::vector< std::vector< std::vector<double> > >
labeled_matrices_vector_from_csv(
        std::istream &in, std::vector<std::string> *labels, size_t dim = 0);

flann::Matrix<double> matrix_from_csv(std::istream &in);
flann::Matrix<double>* matrices_from_csv(std::istream &in, size_t &n);
flann::Matrix<double>* labeled_matrices_from_csv(
        std::istream &in, size_t &n, std::vector<std::string> &labels);

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
