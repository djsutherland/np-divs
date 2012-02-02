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
