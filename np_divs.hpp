#ifndef NP_DIVS_HPP_
#define NP_DIVS_HPP_
#include "basics.hpp"

#include <vector>
#include <flann/flann.hpp>
#include <Eigen/Core>
#include <Eigen/StdVector>

#include "div_func.hpp"
#include "div_l2.hpp"
#include "dkn.hpp"

template <typename Derived>
std::vector<
    Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>,
    Eigen::aligned_allocator<
        Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>
    >
>
np_divs(const std::vector<Eigen::DenseBase<Derived> > &bags,
        unsigned int k = 3)
{
    std::vector<DivFunc> div_funcs;
    div_funcs.push_back(DivL2());

    return np_divs(bags, div_funcs, k);
}

template <typename Derived>
std::vector<
    Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>,
    Eigen::aligned_allocator<
        Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>
    >
>
np_divs(const std::vector<Eigen::DenseBase<Derived> > &bags,
        const std::vector<DivFunc> div_funcs,
        unsigned int k = 3)
{
    // TODO - do this in a way that doesn't do unnecessary work
    return np_divs(bags, bags, div_funcs, k);
}

template <typename Derived>
std::vector<
    Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>,
    Eigen::aligned_allocator<
        Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>
    >
>
np_divs(const std::vector<Eigen::DenseBase<Derived> > &x_bags,
        const std::vector<Eigen::DenseBase<Derived> > &y_bags,
        unsigned int k = 3)
{
    std::vector<DivFunc> div_funcs;
    div_funcs.push_back(DivL2());

    return np_divs(x_bags, y_bags, div_funcs, k);
}

template <typename Derived, typename Distance>
void _make_indices(const std::vector<Eigen::DenseBase<Derived> > &bags,
                   flann::Index<Distance>* indices,
                   flann::Matrix<typename Derived::Scalar>** datasets,
                   const flann::IndexParams &index_params,
                   bool build_index=false) {
    /* Makes a flann::Index for each element of bags (but doesn't build it
     * unless build_index is true).
     * Be sure to free the contents of the arrays appropriately when done:
     * do a delete[] on everything in datasets as well as the indices.
     */
    using Eigen::Dynamic; using Eigen::RowMajor;

    typedef typename Derived::Scalar Scalar;
    typedef Eigen::Matrix<Scalar, Dynamic, Dynamic, RowMajor> RowMajorMatrix;
    typedef flann::Matrix<Scalar> FlannMatrix;
    typedef flann::Index<Distance> FlannIndex;

    size_t num = bags.size();

    for (size_t i = 0; i < num; i++) {
        // need to copy the data into a flann::Matrix
        // can't just use it in-place because we need to be sure it's row-major
        // TODO: check if it's already row-major, and if so, don't copy
        //       (but then figure out how to clear memory...)
        const RowMajorMatrix *m = new RowMajorMatrix(bags[i]);
        const FlannMatrix *data =
            new FlannMatrix(m->data(), m->rows(), m->cols(), m->innerStride());

        FlannIndex* index = new FlannIndex(*data, index_params);
        if (build_index)
            index->buildIndex();

        datasets[i] = data;
        indices[i] = index;
    }
}

template <typename Derived>
std::vector<
    Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>,
    Eigen::aligned_allocator<
        Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>
    >
>
np_divs(const std::vector<Eigen::DenseBase<Derived> > &x_bags,
        const std::vector<Eigen::DenseBase<Derived> > &y_bags,
        const std::vector<DivFunc> div_funcs,
        unsigned int k = 3)
{   /* Calculates the matrix of divergences between x_bags and y_bags for
     * each of the passed div_funcs. Returns a vector of matrices corresponding
     * to the passed div_funcs. Rows of each matrix are an x_bag, columns are
     * a y_bag.
     */

    // TODO - tune the flann stuffs -- just use autotuning?

    using namespace Eigen;
    using std::vector;

    typedef typename Derived::Scalar Scalar;
    typedef flann::Matrix<Scalar> FlannMatrix;
    typedef Matrix<Scalar, Dynamic, Dynamic> EigenMatrix;
    typedef vector<EigenMatrix, aligned_allocator<EigenMatrix> > VecEigMatrix;
    typedef flann::Index<flann::L2<Scalar> > FlannIndex;

    int num_x = x_bags.size();
    int num_y = y_bags.size();
    int num_dfs = div_funcs.size();
    int dim = x_bags[0].cols();

    const flann::KDTreeSingleIndexParams index_params;
    const flann::SearchParams search_params(64);


    // build flann::Index objects for the bags
    FlannIndex* x_indices = new FlannIndex[num_x];
    FlannMatrix** x_datasets = new FlannMatrix*[num_x];
    _make_indices(x_bags, x_indices, x_datasets, index_params);

    FlannIndex* y_indices = new FlannIndex[num_y];
    FlannMatrix** y_datasets = new FlannMatrix*[num_y];
    _make_indices(y_bags, y_indices, y_datasets, index_params);


    // make rhos for the bags
    vector<vector<Scalar> > x_rhos;
    x_rhos.reserve(num_x);
    for (size_t i = 0; i < num_x; i++) {
        x_rhos.push_back(DKN(x_indices[i], x_bags[i], k+1, search_params));
    }

    vector<vector<Scalar> > y_rhos;
    y_rhos.reserve(num_y);
    for (size_t j = 0; j < num_y; j++) {
        y_rhos.push_back(DKN(y_indices[j], y_bags[j], k+1, search_params));
    } // TODO - check that we actually need the y_rhos


    // build the result matrices to fill up
    VecEigMatrix results;
    results.reserve(div_funcs.size());
    for (size_t df = 0; df < num_dfs; df++) {
        results.push_back(EigenMatrix(num_x, num_y));
    }


    // compute the divergences!
    // TODO: threading
    Eigen::DenseBase<Derived> x_bag, y_bag;
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
                results[df](i, j) = div_funcs[df](
                        rho_x, nu_x, rho_y, nu_y, dim, k);
            }
        }
    }

    // think about:
    //    how should threads be split up?
    //    use plain threads (boost?) or TBB?
    //    precompute the necessary stuff, or do locking to compute as needed?


    // TODO: delete[] everything we allocated
    for (size_t i = 0; i < num_x; i++) {
        delete[] x_datasets[i]->ptr();
        delete[] x_indices[i];
    }
    delete[] x_datasets;
    delete[] x_indices;

    for (size_t j = 0; j < num_y; j++) {
        delete[] y_datasets[j]->ptr();
        delete[] y_indices[j];
    }
    delete[] y_datasets;
    delete[] y_indices;

    return results;
}

#endif
