#ifndef DKN_HPP_
#define DKN_HPP_
#include "basics.hpp"

#include <cmath>
#include <vector>
#include <flann/flann.hpp>

template <typename Distance>
std::vector<typename Distance::ResultType> DKN(
        flann::Index<Distance> &index,
        const flann::Matrix<typename Distance::ElementType> &query,
        int k = 3,
        const flann::SearchParams &search_params = flann::SearchParams(),
        bool take_sqrt = true)
{   /* Get the distances to the k-th nearest neighbor of each element in query
     * using the passed index and search params.
     *
     * Make sure that the passed index objects have already done buildIndex().
     *
     * Because flann::L2 is actually the squared Euclidean distance, this
     * function by default square-roots the results. Pass take_sqrt=false to
     * avoid this.
     * */
    using std::vector;
    using flann::Matrix;

    typedef typename Distance::ResultType DistanceType;

    // matrices to store the results
    Matrix<int> indices(new int[query.rows*k], query.rows, k);
    Matrix<DistanceType> dists(new DistanceType[query.rows*k], query.rows, k);

    // search!
    index.knnSearch(query, indices, dists, k, search_params);

    // get out just the results we want
    vector<DistanceType> dkn;
    dkn.reserve(query.rows);
    if (take_sqrt)
        for (size_t i = 0; i < query.rows; i++)
            dkn.push_back(std::sqrt(dists[i][k-1]));
    else
        for (size_t i = 0; i < query.rows; i++)
            dkn.push_back(dists[i][k-1]);

    delete[] indices.ptr();
    delete[] dists.ptr();

    return dkn;
}

#endif
