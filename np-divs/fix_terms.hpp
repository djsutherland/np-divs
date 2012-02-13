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
#ifndef NPDIVS_FIX_TERMS_HPP_
#define NPDIVS_FIX_TERMS_HPP_
#include "np-divs/basics.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace npdivs {

template <typename T>
bool cmp_with_inf(T i, T j) {
    if (std::isinf(i)) return true;
    else if (std::isinf(j)) return false;
    else return i < j;
}

template <typename T>
struct greater_than {
    greater_than(T x) : base(x) {}
    bool operator()(T y) { return y > base; }

    private:
        T base;
};

template <typename T>
T quantile(std::vector<T> &vec, double p) {
    /* Finds the p-th quantile of vec, changing its order in doing so
     * (unless it's already sorted). Assumes that vec contains no nan
     * values.
     *
     * Like the MATLAB quantile() function, but for only a single p.
     * Specifically:
     *
     * Quantiles are specified using cumulative probabilities from 0 to 1. For
     * an n-element vector vec, quantile computes quantiles as follows:
     *
     * 1. The sorted values in vec are taken as the
     *      (0.5/n), (1.5/n), ..., ([n–0.5]/n) quantiles.
     *
     * 2. Linear interpolation is used to compute quantiles for probabilities
     *      between (0.5/n) and ([n–0.5]/n).
     *
     * 3. The minimum or maximum values in vec are assigned to quantiles for
     *      probabilities outside that range.
     */
    typedef typename std::vector<T>::size_type sz;

    sz n = vec.size();

    if (p > (n-.5) / n) {
        return *std::max_element(vec.begin(), vec.end());

    } else if (p < .5 / n) {
        return *std::min_element(vec.begin(), vec.end());

    } else {
        // find the bounds we're interpolating between
        // that is, find i s.t. (i+.5) / n <= p < (i+1.5)/n
        double t = n * p - .5;
        sz i = (sz) std::floor(t);

        // partial sort so that the ith element is in vec[i], with
        // smaller indices below it and larger above
        std::nth_element(vec.begin(), vec.begin() + i, vec.end());

        if (i == t) {
            // did we luck out and get an integer index?
            return vec[i];

        } else {
            // do linear interpolation between this and next index
            T smaller = vec[i];

            // figure out what next index is
            T larger = *std::min_element(vec.begin() + i + 1, vec.end());

            // interpolate
            return smaller + (larger - smaller) * (t - i);
        }
    }
}

template <typename T>
void fix_terms(std::vector<T> &terms, double ub = .99) {
    /* Takes a vector of elements and replaces any infinite or very-large
     * elements with the value of the highest non-very-large element, as well
     * as throwing away any nan values, possibly changing the order.
     * "Very-large" is defined as the ub-th quantile if ub < 1, otherwise the
     * largest non-inf element. Note that values of -inf are not altered.
     */
    typedef typename std::vector<T>::size_type sz;

    using std::max_element;
    using std::nth_element;
    using std::replace_if;
    using std::remove_if;

    T cutoff;
    bool find_noninf_max = true;

    // throw away any nans
    terms.erase(remove_if(terms.begin(), terms.end(), std::isnan<T>),
                terms.end());

    // try finding the ub-th percentile
    if (ub < 1) {
        cutoff = quantile(terms, ub);
        find_noninf_max = std::isinf(cutoff) || std::isnan(cutoff);
    }

    // just use the highest non-inf element
    if (find_noninf_max) {
        cutoff = *max_element(terms.begin(), terms.end(), cmp_with_inf<T>);
    }

    // replace anything greater than cutoff with cutoff
    replace_if(terms.begin(), terms.end(), greater_than<T>(cutoff), cutoff);
}

}

#endif
