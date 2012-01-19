#ifndef NP_DIVS_HPP_
#define NP_DIVS_HPP_
#include "basics.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include <boost/format.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <flann/flann.hpp>

#include "div_func.hpp"
#include "div_l2.hpp"
#include "dkn.hpp"

namespace NPDivs {

template <typename Scalar>
void np_divs(
        const flann::Matrix<Scalar> *bags, size_t num_bags,
        flann::Matrix<Scalar>* results,
        int k = 3)
{
    return np_divs(bags, num_bags, NULL, num_bags, results, k);
}

template <typename Scalar>
void np_divs(
        const flann::Matrix<Scalar> *bags, size_t num_bags,
        const boost::ptr_vector<DivFunc> &div_funcs,
        flann::Matrix<Scalar>* results,
        int k = 3)
{
    return np_divs(bags, num_bags, (flann::Matrix<Scalar>*) NULL, num_bags,
            div_funcs, results, k);
}

template <typename Scalar>
void np_divs(
        const flann::Matrix<Scalar> *x_bags, size_t num_x,
        const flann::Matrix<Scalar> *y_bags, size_t num_y,
        flann::Matrix<Scalar>* results,
        int k = 3)
{
    boost::ptr_vector<DivFunc> div_funcs;
    div_funcs.push_back(new DivL2());

    return np_divs(x_bags, num_x, y_bags, num_y, div_funcs, results, k);
}


// TODO: np_divs overload that writes into vector<vector<vector<float> > >

template <typename Scalar>
void np_divs(
    const flann::Matrix<Scalar>* x_bags, size_t num_x,
    const flann::Matrix<Scalar>* y_bags, size_t num_y,
    const boost::ptr_vector<DivFunc> &div_funcs,
    flann::Matrix<float>* results,
    int k = 3,
    const flann::IndexParams &index_params = flann::KDTreeSingleIndexParams(),
    const flann::SearchParams &search_params = flann::SearchParams(64),
    bool verify_results_alloced = true)
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

    using std::transform;
    using std::vector;

    typedef flann::L2<Scalar> Distance;

    typedef flann::Matrix<Scalar> Matrix;
    typedef flann::Index<Distance> Index;
    typedef vector<typename Distance::ResultType> DistVec;

    int num_dfs = div_funcs.size();
    int dim = x_bags[0].cols;

    bool bags_are_same = y_bags == NULL || x_bags == y_bags;

    // check that result matrices are allocated properly
    if (verify_results_alloced) {
        for (size_t i = 0; i < div_funcs.size(); i++) {
            flann::Matrix<float> m = results[i];
            if (m.rows != num_x || m.cols != num_y) {
                throw std::length_error(
                    (boost::format("expected matrix %d to be %dx%d; it's %d%d")
                     % i % num_x % num_y % m.rows % m.cols).str()
                );
            }
        }
    }

    // build flann::Index objects for the bags
    Index** x_indices = (Index**) malloc(sizeof(Index*) * num_x);
    Index** y_indices;

    for (size_t i = 0; i < num_x; i++) {
        Index* idx = new Index(x_bags[i], index_params);
        idx->buildIndex();
        x_indices[i] = idx;
    }

    if (!bags_are_same) {
        y_indices = (Index**) malloc(sizeof(Index*) * num_y);
        for (size_t i = 0; i < num_y; i++) {
            Index* idx = new Index(y_bags[i], index_params);
            idx->buildIndex();
            y_indices[i] = idx;
        }
    }

    // make rhos for the bags
    // TODO - check that we actually need the y_rhos
    vector<DistVec> x_rhos, y_rhos;
    x_rhos.reserve(num_x);
    for (size_t i = 0; i < num_x; i++)
        x_rhos.push_back(DKN(*x_indices[i], x_bags[i], k+1, search_params));

    if (!bags_are_same) {
        y_rhos.reserve(num_y);
        for (size_t j = 0; j < num_y; j++)
            y_rhos.push_back(DKN(*y_indices[j], y_bags[j], k+1, search_params));
    }

    // compute the divergences!  // TODO: threading
    DistVec nu_x, nu_y;

    // TODO - check that we actually need nu_y

    if (!bags_are_same) {
        for (size_t i = 0; i < num_x; i++) {
            DistVec &rho_x = x_rhos[i];
            for (size_t j = 0; j < num_y; j++) {
                DistVec &rho_y = y_rhos[j];

                DistVec nu_x = DKN(*y_indices[j], x_bags[i], k, search_params);
                nu_y = DKN(*x_indices[i], y_bags[j], k, search_params);

                for (size_t df = 0; df < num_dfs; df++) {
                    results[df][i][j] = div_funcs[df](
                            rho_x, nu_x, rho_y, nu_y, dim, k);
                }
            }
        }
    } else {
        // we can save some nearest-neighbor calculation by calculating both
        // directions at the same time
        for (size_t i = 0; i < num_x; i++) {
            DistVec &rho_x = x_rhos[i];

            // compare with self
            for (size_t df = 0; df < num_dfs; df++) {
                results[df][i][i] = div_funcs[df](
                        rho_x, rho_x, rho_x, rho_x, dim, k);
            }

            // compare with others
            for (size_t j = 0; j <= i; j++) {
                DistVec &rho_y = x_rhos[j];

                nu_x = DKN(*x_indices[j], x_bags[i], k, search_params);
                nu_y = DKN(*x_indices[i], x_bags[j], k, search_params);

                for (size_t df = 0; df < num_dfs; df++) {
                    results[df][i][j] = div_funcs[df](
                            rho_x, nu_x, rho_y, nu_y, dim, k);
                    results[df][j][i] = div_funcs[df](
                            rho_y, nu_y, rho_x, nu_x, dim, k);
                }
            }
        }
    }

    // think about:
    //    how should threads be split up?
    //    use plain threads (boost?) or TBB?
    //    precompute the necessary stuff, or do locking to compute as needed?

    for (size_t i = 0; i < num_x; i++)
        delete x_indices[i];
    free(x_indices);

    if (!bags_are_same) {
        for (size_t i = 0; i < num_y; i++)
            delete y_indices[i];
        free(y_indices);
    }
}

}
#endif
