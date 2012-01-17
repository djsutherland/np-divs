#ifndef DKN_HPP_
#define DKN_HPP_
#include "basics.hpp"

#include <vector>
#include <flann/flann.hpp>

template <typename Distance>
std::vector<typename Distance::ResultType> DKN(
        flann::Index<Distance> &index,
        const flann::Matrix<typename Distance::ElementType> &query,
        unsigned int k = 3,
        const flann::SearchParams &search_params = flann::SearchParams())
{   /* Get the distances to the k-th nearest neighbor of each element in query
     * using the passed index and search params.
     */
    using std::vector;
    using flann::Matrix;

    typedef typename Distance::ResultType DistanceType;

    // matrices to store the results
    Matrix<int> indices(new int[query.rows*k], query.rows, k);
    Matrix<DistanceType> dists(new float[query.rows*k], query.rows, k);

    // build the index if necessary
    index.buildIndex();

    // search!
    index.knnSearch(query, indices, dists, k, search_params);

    // get out just the results we want
    vector<DistanceType> dkn;
    dkn.reserve(query.rows);
    for (size_t i = 0; i < query.rows; i++) {
        dkn.push_back(query[i][k]);
    }

    delete[] indices.ptr();
    delete[] dists.ptr();

    return dkn;
}

#endif
