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
#ifndef NPDIVS_MATRIX_ARRAYS_HPP_
#define NPDIVS_MATRIX_ARRAYS_HPP_
#include "np-divs/basics.hpp"

#include <stdexcept>

#include <boost/throw_exception.hpp>

#include <flann/util/matrix.h>

namespace npdivs {

////////////////////////////////////////////////////////////////////////////////
// Helper functions for allocating/freeing matrix arrays

template <typename Scalar>
flann::Matrix<Scalar>* alloc_matrix_array(size_t n, size_t rows, size_t cols);

template <typename Scalar>
void free_matrix_array(flann::Matrix<Scalar> *array, size_t n);


////////////////////////////////////////////////////////////////////////////////
// Helper functions for converting stacked vectors to matrices

template <typename Scalar>
flann::Matrix<Scalar> vector_to_matrix(std::vector< std::vector<Scalar> > vec);

template <typename Scalar>
flann::Matrix<Scalar>* vector_to_matrix_array(
        std::vector< std::vector< std::vector<Scalar> > > vec);

////////////////////////////////////////////////////////////////////////////////
// Template implementations

template <typename Scalar>
flann::Matrix<Scalar>* alloc_matrix_array(size_t n, size_t rows, size_t cols) {
    typedef flann::Matrix<Scalar> Matrix;
    size_t s = rows * cols;

    Matrix* array = new Matrix[n];
    for (size_t i = 0; i < n; i++)
        array[i] = Matrix(new Scalar[s], rows, cols);
    return array;
}

template <typename Scalar>
void free_matrix_array(flann::Matrix<Scalar> *array, size_t n) {
    for (size_t i = 0; i < n; i++)
        delete[] array[i].ptr();
    delete[] array;
}

template <typename Scalar>
flann::Matrix<Scalar> vector_to_matrix(std::vector<std::vector<Scalar> > vec) {
    typedef flann::Matrix<Scalar> Matrix;

    // check dimensions
    size_t rows = vec.size();
    if (rows == 0)
        BOOST_THROW_EXCEPTION(std::domain_error("can't convert empty vector"));

    size_t cols = vec[0].size();
    for (size_t i = 0; i < rows; i++) {
        if (vec[i].size() != cols)
            BOOST_THROW_EXCEPTION(std::domain_error(
                        "can't make a nonrectangular matrix"));
    }

    // make the matrix
    Matrix mat = Matrix(new Scalar[rows * cols], rows, cols);
    Scalar* data = mat.ptr();

    // copy data in
    for (size_t i = 0; i < rows; i++) {
        copy(vec[i].begin(), vec[i].end(), data + i*cols);
    }

    return mat;
}

template <typename Scalar>
flann::Matrix<Scalar>* vector_to_matrix_array(
        std::vector< std::vector< std::vector<Scalar> > > vec)
{
    typedef flann::Matrix<Scalar> Matrix;

    size_t n = vec.size();
    if (n == 0)
        BOOST_THROW_EXCEPTION(std::domain_error("can't convert empty vector"));

    Matrix* array = new Matrix[n];
    size_t i;
    try {
        for (i = 0; i < n; i++)
            array[i] = vector_to_matrix(vec[i]);

    } catch (std::domain_error &e) {
        for (size_t j = 0; j < i; j++)
            delete[] array[j].ptr();
        delete[] array;
        throw;
    }

    return array;
}

}

#endif
