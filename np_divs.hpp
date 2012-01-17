#ifndef NP_DIVS_HPP_
#define NP_DIVS_HPP_
#include "basics.hpp"

#include <algorithm>
#include <vector>

#include <boost/ptr_container/ptr_vector.hpp>
#include <flann/flann.hpp>

#include "div_func.hpp"
#include "div_l2.hpp"
#include "dkn.hpp"

template <typename Scalar>
std::vector<flann::Matrix<Scalar> > np_divs(
        const std::vector<flann::Matrix<Scalar> > &bags,
        unsigned int k = 3)
{
    boost::ptr_vector<DivFunc> div_funcs;
    div_funcs.push_back(new DivL2());

    return np_divs(bags, div_funcs, k);
}

template <typename Scalar>
std::vector<flann::Matrix<Scalar> > np_divs(
        const std::vector<flann::Matrix<Scalar> > &bags,
        const boost::ptr_vector<DivFunc> div_funcs,
        unsigned int k = 3)
{
    // TODO - do this in a way that doesn't do unnecessary work
    return np_divs(bags, bags, div_funcs, k);
}

template <typename Scalar>
std::vector<flann::Matrix<Scalar> > np_divs(
        const std::vector<flann::Matrix<Scalar> > &x_bags,
        const std::vector<flann::Matrix<Scalar> > &y_bags,
        unsigned int k = 3)
{
    boost::ptr_vector<DivFunc> div_funcs;
    div_funcs.push_back(new DivL2());

    return np_divs(x_bags, y_bags, div_funcs, k);
}


template <typename Distance>
class _index_builder {
    flann::IndexParams index_params;

    public:
    _index_builder(const flann::IndexParams &index_params) :
        index_params(index_params) {}

    flann::Index<Distance> operator()(
            const flann::Matrix<typename Distance::ElementType> &dataset)
    const {
        return flann::Index<Distance>(dataset, index_params);
    }
};

template <typename Scalar>
std::vector<flann::Matrix<Scalar> > np_divs(
    const std::vector<flann::Matrix<Scalar> > &x_bags,
    const std::vector<flann::Matrix<Scalar> > &y_bags,
    const boost::ptr_vector<DivFunc> div_funcs,
    unsigned int k = 3,
    const flann::IndexParams &index_params = flann::KDTreeSingleIndexParams(),
    const flann::SearchParams &search_params = flann::SearchParams(64))
{   /* Calculates the matrix of divergences between x_bags and y_bags for
     * each of the passed div_funcs. Returns a vector of matrices corresponding
     * to the passed div_funcs. Rows of each matrix are an x_bag, columns are
     * a y_bag.
     *
     * It is the caller's responsibility to explicitly delete[] the .ptr()
     * of each matrix before the returned vector is deallocated.
     * TODO: figure out a wrapper to automatically deallocate
     */

    // TODO - tune the flann indices -- just use autotuning?

    using std::transform;
    using std::vector;

    typedef flann::L2<Scalar> Distance;
    typedef flann::Matrix<Scalar> Matrix;

    int num_x = x_bags.size();
    int num_y = y_bags.size();
    int num_dfs = div_funcs.size();
    int dim = x_bags[0].cols;


    // build flann::Index objects for the bags
    vector<flann::Index<Distance> > x_indices, y_indices;
    const _index_builder<Distance> index_builder(index_params);

    x_indices.reserve(num_x);
    transform(x_bags.begin(), x_bags.end(), x_indices.begin(), index_builder);

    y_indices.reserve(num_y);
    transform(y_bags.begin(), y_bags.end(), y_indices.begin(), index_builder);


    // make rhos for the bags
    // TODO - check that we actually need the y_rhos
    vector<vector<Scalar> > x_rhos, y_rhos;

    x_rhos.reserve(num_x);
    for (size_t i = 0; i < num_x; i++)
        x_rhos.push_back(DKN(x_indices[i], x_bags[i], k+1, search_params));

    y_rhos.reserve(num_y);
    for (size_t j = 0; j < num_y; j++)
        y_rhos.push_back(DKN(y_indices[j], y_bags[j], k+1, search_params));


    // build the result matrices to fill up
    vector<Matrix> results;
    results.reserve(num_dfs);
    for (size_t df = 0; df < num_dfs; df++) {
        results.push_back(Matrix(new Scalar[num_x * num_y], num_x, num_y));
    }


    // compute the divergences!
    // TODO: threading
    Matrix x_bag, y_bag;
    vector<Scalar> rho_x, nu_x, rho_y, nu_y;

    for (size_t i = 0; i < num_x; i++) {
        x_bag = x_bags[i];
        rho_x = x_rhos[i];
        for (size_t j = 0; j < num_y; j++) {
            y_bag = y_bags[j];
            rho_y = y_rhos[j];

            nu_x = DKN(x_indices[i], y_bags[j], k, search_params);
            nu_y = DKN(y_indices[j], x_bags[i], k, search_params);
            // TODO - check that we actually need nu_y

            for (size_t df = 0; df < num_dfs; df++) {
                results[df][i][j] = div_funcs[df](
                        rho_x, nu_x, rho_y, nu_y, dim, k);
            }
        }
    }

    // think about:
    //    how should threads be split up?
    //    use plain threads (boost?) or TBB?
    //    precompute the necessary stuff, or do locking to compute as needed?

    return results;
}

#endif
