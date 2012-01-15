#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <algorithm>
#include <cmath>
#include <math.h>
#include <iterator>
#include <vector>

#include <Eigen/Dense>

// code in here because templates are silly :/

template <typename Derived>
std::vector<typename Derived::Scalar> as_vector(const Eigen::DenseBase<Derived> &v) {
    /* Takes an Eigen vector/matrix/array of values and returns a std::vector
     * with the same contents.
     */
    typedef typename Derived::Scalar Scalar;
    typedef typename std::vector<Scalar>::size_type sz;

    std::vector<Scalar> vec;
    vec.reserve(v.size());

    for (sz i = 0; i < v.size(); i++)
        vec.push_back(v(i));

    return vec;
}

template <typename T>
bool cmp_with_inf(T i, T j) { return std::isinf(i) || i < j; }

template <typename T>
struct greater_than {
    greater_than(T x) : base(x) {}
    bool operator()(T y) { return y > base; }

    private:
        T base;
};

template <typename T>
void fix_terms(std::vector<T> &terms, double ub = .99) {
    /* Takes a vector of elements and replaces any infinite or very-large
     * elements with the value of the highest non-very-large element, as well
     * as possibly changing the order. "Very-large" is defined as the ub-th
     * quantile if ub < 1, otherwise the largest non-inf element.
     */
    typedef typename std::vector<T>::size_type sz;


    T cutoff;
    bool find_noninf_max = true;

    // try finding the ub-th percentile
    if (ub < 1) {
        sz k = terms.size() * ub; // the index we want
        std::nth_element(terms.begin(), terms.begin() + k, terms.end());
        cutoff = terms[k];

        find_noninf_max = std::isinf(cutoff);
    }

    // just use the highest non-inf element
    if (find_noninf_max) {
        cutoff = *std::max_element(terms.begin(), terms.end(), cmp_with_inf<T>);
    }

    // replace anything greater than cutoff with cutoff
    std::replace_if(terms.begin(), terms.end(), greater_than<T>(cutoff), cutoff);
}


#endif
