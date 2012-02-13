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
#ifndef NPDIVS_DKN_HPP_
#define NPDIVS_DKN_HPP_
#include "np-divs/basics.hpp"

#include <cmath>
#include <vector>
#include <flann/flann.hpp>

namespace npdivs {

template <typename Distance, typename ResultType>
std::vector<ResultType> DKN(
        flann::Index<Distance> &index,
        const flann::Matrix<typename Distance::ElementType> &query,
        int k = 3,
        const flann::SearchParams &search_params = flann::SearchParams(),
        bool take_sqrt = true)
{   /* Get the distances to the k-th nearest neighbor of each element in query
     * using the passed index and search params.
     *
     * Make sure that the passed index objects have already done buildIndex().
     * Since FLANN searches are thread-safe, so is this function.
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
    vector<ResultType> dkn;
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


template <typename Distance>
std::vector<typename Distance::ResultType> DKN(
        flann::Index<Distance> &index,
        const flann::Matrix<typename Distance::ElementType> &query,
        int k = 3,
        const flann::SearchParams &search_params = flann::SearchParams(),
        bool take_sqrt = true)
{
    return DKN<Distance, typename Distance::ResultType>(
            index, query, k, search_params, take_sqrt);
}

}

#endif
