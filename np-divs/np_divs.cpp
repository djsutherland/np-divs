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
#include "np-divs/np_divs.hpp"

namespace NPDivs{

// explicit instantiations for np_divs() overloads with doubles

template void np_divs(
    const flann::Matrix<double> *bags, size_t num_bags,
    flann::Matrix<double> *results,
    int k,
    const flann::IndexParams &index_params,
    const flann::SearchParams &search_params,
    size_t num_threads,
    bool verify_results_alloced);

template void np_divs(
    const flann::Matrix<double> *bags, size_t num_bags,
    const boost::ptr_vector<DivFunc> &div_funcs,
    flann::Matrix<double> *results,
    int k,
    const flann::IndexParams &index_params,
    const flann::SearchParams &search_params,
    size_t num_threads,
    bool verify_results_alloced);

template void np_divs(
    const flann::Matrix<double> *x_bags, size_t num_x,
    const flann::Matrix<double> *y_bags, size_t num_y,
    flann::Matrix<double>* results,
    int k,
    const flann::IndexParams &index_params,
    const flann::SearchParams &search_params,
    size_t num_threads,
    bool verify_results_alloced);

template void np_divs(
    const flann::Matrix<double> *x_bags, size_t num_x,
    const flann::Matrix<double> *y_bags, size_t num_y,
    const boost::ptr_vector<DivFunc> &div_funcs,
    flann::Matrix<double>* results,
    int k,
    const flann::IndexParams &index_params,
    const flann::SearchParams &search_params,
    size_t num_threads,
    bool verify_results_alloced);


// explicit instantiations for np_divs() overloads with floats

template void np_divs(
    const flann::Matrix<float> *bags, size_t num_bags,
    flann::Matrix<double> *results,
    int k,
    const flann::IndexParams &index_params,
    const flann::SearchParams &search_params,
    size_t num_threads,
    bool verify_results_alloced);

template void np_divs(
    const flann::Matrix<float> *bags, size_t num_bags,
    const boost::ptr_vector<DivFunc> &div_funcs,
    flann::Matrix<double> *results,
    int k,
    const flann::IndexParams &index_params,
    const flann::SearchParams &search_params,
    size_t num_threads,
    bool verify_results_alloced);

template void np_divs(
    const flann::Matrix<float> *x_bags, size_t num_x,
    const flann::Matrix<float> *y_bags, size_t num_y,
    flann::Matrix<double>* results,
    int k,
    const flann::IndexParams &index_params,
    const flann::SearchParams &search_params,
    size_t num_threads,
    bool verify_results_alloced);

template void np_divs(
    const flann::Matrix<float> *x_bags, size_t num_x,
    const flann::Matrix<float> *y_bags, size_t num_y,
    const boost::ptr_vector<DivFunc> &div_funcs,
    flann::Matrix<double>* results,
    int k,
    const flann::IndexParams &index_params,
    const flann::SearchParams &search_params,
    size_t num_threads,
    bool verify_results_alloced);

} // end namespace
