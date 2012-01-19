#ifndef NP_DIVS_HPP_
#define NP_DIVS_HPP_
#include "basics.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/format.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <flann/flann.hpp>

#include "div_func.hpp"
#include "div_l2.hpp"
#include "dkn.hpp"

namespace NPDivs {

////////////////////////////////////////////////////////////////////////////////
// Some functions that are helpful for allocating/freeing matrix arrays

template <typename Scalar>
flann::Matrix<Scalar>* alloc_matrix_array(size_t n, size_t rows, size_t cols);

template <typename Scalar>
void free_matrix_array(flann::Matrix<Scalar> *array, size_t n);

////////////////////////////////////////////////////////////////////////////////
// Declarations of the main np_divs functions

#define INDEX_PARAMS flann::KDTreeSingleIndexParams()
#define SEARCH_PARAMS flann::SearchParams(64)

template <typename Scalar>
void np_divs(
    const flann::Matrix<Scalar> *bags,
    size_t num_bags,
    flann::Matrix<float> *results,
    int k = 3,
    const flann::IndexParams &index_params = INDEX_PARAMS,
    const flann::SearchParams &search_params = SEARCH_PARAMS,
    bool verify_results_alloced = true);

template <typename Scalar>
void np_divs(
    const flann::Matrix<Scalar> *bags,
    size_t num_bags,
    const boost::ptr_vector<DivFunc> &div_funcs,
    flann::Matrix<float> *results,
    int k = 3,
    const flann::IndexParams &index_params = INDEX_PARAMS,
    const flann::SearchParams &search_params = SEARCH_PARAMS,
    bool verify_results_alloced = true);

template <typename Scalar>
void np_divs(
    const flann::Matrix<Scalar> *x_bags,
    size_t num_x,
    const flann::Matrix<Scalar> *y_bags,
    size_t num_y,
    flann::Matrix<float>* results,
    int k = 3,
    const flann::IndexParams &index_params = INDEX_PARAMS,
    const flann::SearchParams &search_params = SEARCH_PARAMS,
    bool verify_results_alloced = true);

template <typename Scalar>
void np_divs(
    const flann::Matrix<Scalar> *x_bags, size_t num_x,
    const flann::Matrix<Scalar> *y_bags, size_t num_y,
    const boost::ptr_vector<DivFunc> &div_funcs,
    flann::Matrix<float>* results,
    int k = 3,
    const flann::IndexParams &index_params = INDEX_PARAMS,
    const flann::SearchParams &search_params = SEARCH_PARAMS,
    bool verify_results_alloced = true);



////////////////////////////////////////////////////////////////////////////////
// Declarations of helpers used in the code below

template <typename T>
void verify_allocated(
        flann::Matrix<T> *matrices,
        size_t num_matrices, size_t rows, size_t cols);
// throws a std::length_error if they're not the right size

template <typename Distance>
flann::Index<Distance>** make_indices(
        const flann::Matrix<typename Distance::ElementType> *datasets,
        size_t n,
        const flann::IndexParams index_params);

template <typename Distance>
inline void free_indices(flann::Index<Distance>** indices, size_t n);


template <typename Distance>
std::vector<std::vector<typename Distance::ResultType> > get_rhos(
        const flann::Matrix<typename Distance::ElementType> *bags,
        flann::Index<Distance> **indices,
        size_t n,
        int k,
        const flann::SearchParams &search_params = SEARCH_PARAMS,
        bool actually_do_it = true);
// bool arg is a hack so it's easier to define refs that you might not need


////////////////////////////////////////////////////////////////////////////////
// Implementations of the np_divs overloads
template <typename Scalar>
void np_divs(
        const flann::Matrix<Scalar> *bags,
        size_t num_bags,
        flann::Matrix<float>* results,
        int k,
        const flann::IndexParams &index_params,
        const flann::SearchParams &search_params,
        bool verify_results_alloced)
{
    return np_divs(bags, num_bags, NULL, num_bags, results, k,
            index_params, search_params, verify_results_alloced);
}

template <typename Scalar>
void np_divs(
        const flann::Matrix<Scalar> *bags,
        size_t num_bags,
        const boost::ptr_vector<DivFunc> &div_funcs,
        flann::Matrix<float>* results,
        int k,
        const flann::IndexParams &index_params,
        const flann::SearchParams &search_params,
        bool verify_results_alloced)
{
    using std::vector;

    typedef flann::L2<Scalar> Distance;

    typedef flann::Matrix<Scalar> Matrix;
    typedef flann::Index<Distance> Index;
    typedef vector<typename Distance::ResultType> DistVec;

    size_t num_dfs = div_funcs.size();
    size_t dim = bags[0].cols;

    // some setup
    if (verify_results_alloced)
        verify_allocated(results, num_dfs, num_bags, num_bags);

    Index** indices = make_indices<Distance>(bags, num_bags, index_params);

    const vector<DistVec> &rhos =
        get_rhos(bags, indices, num_bags, k, search_params);

    // compute away!
    for (size_t i = 0; i < num_bags; i++) {
        const Matrix &x_bag = bags[i];
        Index &x_index = *indices[i];
        const DistVec &rho_x = rhos[i];


        // calculate the nu from x_bag to itself
        // TODO - is this actually what we want?
        const DistVec &nu_xx = DKN(x_index, x_bag, k, search_params);

        // compare with self
        for (size_t df = 0; df < num_dfs; df++) {
            results[df][i][i] = div_funcs[df](
                    rho_x, nu_xx, rho_x, nu_xx, dim, k);
        }

        // compare with others (both to and from)
        for (size_t j = 0; j < i; j++) {
            const Matrix &y_bag = bags[j];
            Index &y_index = *indices[j];
            const DistVec &rho_y = rhos[j];

            const DistVec nu_x = DKN(
                    y_index, x_bag, k, search_params);
            const DistVec nu_y = DKN(
                    x_index, y_bag, k, search_params);

            for (size_t df = 0; df < num_dfs; df++) {
                results[df][i][j] = div_funcs[df](
                        rho_x, nu_x, rho_y, nu_y, dim, k);
                results[df][j][i] = div_funcs[df](
                        rho_y, nu_y, rho_x, nu_x, dim, k);
            }
        }
    }

    free_indices(indices, num_bags);

}

template <typename Scalar>
void np_divs(
        const flann::Matrix<Scalar> *x_bags, size_t num_x,
        const flann::Matrix<Scalar> *y_bags, size_t num_y,
        flann::Matrix<float>* results,
        int k,
        const flann::IndexParams &index_params,
        const flann::SearchParams &search_params,
        bool verify_results_alloced)
{
    boost::ptr_vector<DivFunc> div_funcs;
    div_funcs.push_back(new DivL2());

    return np_divs(x_bags, num_x, y_bags, num_y, div_funcs, results, k,
            index_params, search_params, verify_results_alloced);
}


// TODO: np_divs overload that writes into vector<vector<vector<float> > >

template <typename Scalar>
void np_divs(
    const flann::Matrix<Scalar>* x_bags, size_t num_x,
    const flann::Matrix<Scalar>* y_bags, size_t num_y,
    const boost::ptr_vector<DivFunc> &div_funcs,
    flann::Matrix<float>* results,
    int k,
    const flann::IndexParams &index_params,
    const flann::SearchParams &search_params,
    bool verify_results_alloced)
{   /* Calculates the matrix of divergences between x_bags and y_bags for
     * each of the passed div_funcs, and writes them into the preallocated
     * array of matrices (div_funcs.size() bags, each with num_x rows and
     * num_y cols) to the passed div_funcs. Rows of each matrix are an x_bag,
     * columns are a y_bag.
     *
     * By default, conducts a quick check that the result matrices were
     * allocated properly; if you're sure that you did and want to skip this
     * check, pass verify_results_alloced=false.
     */

    // TODO - figure out how to tune flann indices (better than autotuning each)

    using std::vector;

    typedef flann::L2<Scalar> Distance;

    typedef flann::Matrix<Scalar> Matrix;
    typedef flann::Index<Distance> Index;
    typedef vector<typename Distance::ResultType> DistVec;

    // are we actually comparing bags with themselves?
    // if so, this overload is somewhat more efficient
    if (y_bags == NULL || y_bags == x_bags)
        return np_divs(x_bags, num_x, div_funcs, results, k,
                index_params, search_params, verify_results_alloced);

    // initial setup work
    size_t num_dfs = div_funcs.size();
    size_t dim = x_bags[0].cols;

    if (verify_results_alloced)
        verify_allocated(results, num_dfs, num_x, num_y);

    Index** x_indices = make_indices<Distance>(x_bags, num_x, index_params);
    Index** y_indices = make_indices<Distance>(y_bags, num_y, index_params);

    const vector<DistVec> &x_rhos =
            get_rhos(x_bags, x_indices, num_x, k, search_params);
    const vector<DistVec> &y_rhos =
            get_rhos(y_bags, y_indices, num_y, k, search_params);

    // compute the divergences!
    //
    // TODO: threading
    // think about:
    //    how should threads be split up?
    //    use plain threads (boost?) or TBB?
    //    precompute the necessary stuff, or do locking to compute as needed?
    //
    // TODO - check that we actually need nu_y

    for (size_t i = 0; i < num_x; i++) {
        const Matrix &x_bag = x_bags[i];
        Index &x_index = *x_indices[i];
        const DistVec &rho_x = x_rhos[i];

        for (size_t j = 0; j < num_y; j++) {
            const Matrix &y_bag = y_bags[j];
            Index &y_index = *y_indices[j];
            const DistVec &rho_y = y_rhos[j];

            const DistVec &nu_x = DKN(y_index, x_bag, k, search_params);
            const DistVec &nu_y = DKN(x_index, y_bag, k, search_params);

            for (size_t df = 0; df < num_dfs; df++) {
                results[df][i][j] = div_funcs[df](
                        rho_x, nu_x, rho_y, nu_y, dim, k);
            }
        }
    }

    free_indices(x_indices, num_x);
    free_indices(y_indices, num_y);
}

////////////////////////////////////////////////////////////////////////////////
// Implementations of helpers

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

template <typename T>
void verify_allocated(
        flann::Matrix<T> *matrices, size_t num_matrices,
        size_t rows, size_t cols)
{
    for (size_t i = 0; i < num_matrices; i++) {
        const flann::Matrix<T> &m = matrices[i];
        if (m.rows != rows || m.cols != cols) {
            boost::format err =
                boost::format("expected matrix %d to be %dx%d; it's %d%d")
                % i % rows % cols % m.rows % m.cols;
            std::cerr << err << std::endl;
            throw std::length_error(err.str());
        }
    }
}

template <typename Distance>
flann::Index<Distance>** make_indices(
        const flann::Matrix<typename Distance::ElementType> *datasets,
        size_t number,
        const flann::IndexParams index_params)
{
    typedef flann::Index<Distance> Index;

    // malloc to avoid calling constructors
    Index** indices = (Index**) malloc(sizeof(Index*) * number);

    for (size_t i = 0; i < number; i++) {
        Index* idx = new Index(datasets[i], index_params);
        idx->buildIndex();
        indices[i] = idx;
    }

    return indices;
}

template <typename Distance>
void free_indices(flann::Index<Distance>** indices, size_t n) {
    for (size_t i = 0; i < n; i++)
        delete indices[i];
    free(indices);
}


template <typename Distance>
std::vector<std::vector<typename Distance::ResultType> > get_rhos(
        const flann::Matrix<typename Distance::ElementType> *bags,
        flann::Index<Distance> **indices,
        size_t n,
        int k,
        const flann::SearchParams &search_params,
        bool actually_do_it)
{
    std::vector<std::vector<typename Distance::ResultType> > rhos;
    if (actually_do_it) {
        rhos.reserve(n);
        for (size_t i = 0; i < n; i++)
            rhos.push_back(DKN(*indices[i], bags[i], k+1, search_params));
    }
    return rhos;
}

} // close namespace
#endif
