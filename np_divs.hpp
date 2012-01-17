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

template <typename Scalar>
void np_divs(
        const flann::Matrix<Scalar> *bags, size_t num_bags,
        flann::Matrix<Scalar>* results,
        unsigned int k = 3)
{
    boost::ptr_vector<DivFunc> div_funcs;
    div_funcs.push_back(new DivL2());

    return np_divs(bags, num_bags, div_funcs, results, k);
}

template <typename Scalar>
void np_divs(
        const flann::Matrix<Scalar> *bags, size_t num_bags,
        const boost::ptr_vector<DivFunc> &div_funcs,
        flann::Matrix<Scalar>* results,
        unsigned int k = 3)
{
    // TODO - do this in a way that doesn't do unnecessary work
    return np_divs(bags, num_bags, bags, num_bags, div_funcs, results, k);
}

template <typename Scalar>
void np_divs(
        const flann::Matrix<Scalar> *x_bags, size_t num_x,
        const flann::Matrix<Scalar> *y_bags, size_t num_y,
        flann::Matrix<Scalar>* results,
        unsigned int k = 3)
{
    boost::ptr_vector<DivFunc> div_funcs;
    div_funcs.push_back(new DivL2());

    return np_divs(x_bags, num_x, y_bags, num_y, div_funcs, results, k);
}


template <typename Distance>
class _index_builder {
    flann::IndexParams index_params;
    typedef flann::Index<Distance> Index;

    public:
    _index_builder(const flann::IndexParams &index_params) :
        index_params(index_params) {}

    Index operator()(
            const flann::Matrix<typename Distance::ElementType> &dataset)
    const {
        Index idx(dataset, index_params);
        idx.buildIndex();
        return idx;
    }
};

// TODO: np_divs overload that writes into vector<vector<vector<float> > >

template <typename Scalar>
void np_divs(
    const flann::Matrix<Scalar>* &x_bags, size_t num_x,
    const flann::Matrix<Scalar>* &y_bags, size_t num_y,
    const boost::ptr_vector<DivFunc> &div_funcs,
    flann::Matrix<Scalar>* results,
    unsigned int k = 3,
    const flann::IndexParams &index_params = flann::KDTreeSingleIndexParams(),
    const flann::SearchParams &search_params = flann::SearchParams(64))
{   /* Calculates the matrix of divergences between x_bags and y_bags for
     * each of the passed div_funcs, and writes them into the preallocated
     * array of matrices (div_funcs.size() bags, each with num_x rows and
     * num_y cols) to the passed div_funcs. Rows of each matrix are an x_bag,
     * columns are a y_bag.
     */

    // TODO - tune the flann indices -- just use autotuning?

    using std::transform;
    using std::vector;

    typedef flann::L2<Scalar> Distance;
    typedef flann::Matrix<Scalar> Matrix;
    typedef flann::Index<Distance> Index;

    int num_dfs = div_funcs.size();
    int dim = x_bags[0].cols;


    // check that result matrices are allocated properly
    for (size_t i = 0; i < div_funcs.size(); i++) {
        Matrix m = results[i];
        if (m.rows != num_x || m.cols != num_y) {
            throw std::length_error(
                (boost::format("expected matrix %d to be %dx%d; it's %d%d")
                 % i % num_x % num_y % m.rows % m.cols).str()
            );
        }
    }


    // build flann::Index objects for the bags
    // const _index_builder<Distance> index_builder(index_params);

    Index** x_indices = (Index**) malloc(sizeof(Index*) * num_x);
    for (size_t i = 0; i < num_x; i++) {
        Index* idx = new Index(x_bags[i], index_params);
        idx->buildIndex();
        x_indices[i] = idx;
    }
    //transform(&x_bags[0], &x_bags[num_x], &x_indices[0], index_builder);

    Index** y_indices = (Index**) malloc(sizeof(Index*) * num_y);
    for (size_t i = 0; i < num_y; i++) {
        Index* idx = new Index(y_bags[i], index_params);
        idx->buildIndex();
        y_indices[i] = idx;
    }
    //transform(&y_bags[0], &y_bags[num_y], &y_indices[0], index_builder);


    // make rhos for the bags
    // TODO - check that we actually need the y_rhos
    vector<vector<Scalar> > x_rhos, y_rhos;

    x_rhos.reserve(num_x);
    for (size_t i = 0; i < num_x; i++)
        x_rhos.push_back(DKN(*x_indices[i], x_bags[i], k+1, search_params));

    y_rhos.reserve(num_y);
    for (size_t j = 0; j < num_y; j++)
        y_rhos.push_back(DKN(*y_indices[j], y_bags[j], k+1, search_params));


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

            nu_x = DKN(*x_indices[i], y_bags[j], k, search_params);
            nu_y = DKN(*y_indices[j], x_bags[i], k, search_params);
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

    for (size_t i = 0; i < num_x; i++)
        delete x_indices[i];
    for (size_t i = 0; i < num_y; i++)
        delete y_indices[i];
    free(x_indices);
    free(y_indices);
}

#endif
