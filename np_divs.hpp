#ifndef NP_DIVS_HPP_
#define NP_DIVS_HPP_

#include <vector>
#include <flann/flann.hpp>
#include <Eigen/Core>

#include "div_func.hpp"

template <typename Derived>
std::vector<Eigen::MatrixXf> np_divs(
        std::vector<Eigen::DenseBase<Derived> > bags,
        unsigned int k = 3)
{
    std::vector<DivFunc> div_funcs;
    div_funcs.push_back(DivL2());

    return np_divs(bags, div_funcs, k);
}

template <typename Derived>
std::vector<Eigen::MatrixXf> np_divs(
        std::vector<Eigen::DenseBase<Derived> > bags,
        std::vector<DivFunc> div_funcs,
        unsigned int k = 3)
{
    // TODO - do this in a way that doesn't do unnecessary work
    return np_divs(bags, bags, div_funcs, k);
}

template <typename Derived>
std::vector<Eigen::MatrixXf> np_divs(
        std::vector<Eigen::DenseBase<Derived> > x_bags,
        std::vector<Eigen::DenseBase<Derived> > y_bags,
        unsigned int k = 3)
{
    std::vector<DivFunc> div_funcs;
    div_funcs.push_back(DivL2());

    return np_divs(x_bags, y_bags, div_funcs, k);
}


template <typename Derived>
std::vector<Eigen::MatrixXf> np_divs(
        std::vector<Eigen::DenseBase<Derived> > x_bags,
        std::vector<Eigen::DenseBase<Derived> > y_bags,
        std::vector<DivFunc> div_funcs,
        unsigned int k = 3)
{   /* Calculates the matrix of divergences between x_bags and y_bags for
     * each of the passed div_funcs. Returns a vector of matrices corresponding
     * to the passed div_funcs. Rows of each matrix are an x_bag, columns are
     * a y_bag.
     */

    // is there a nice way to use an Eigen::Matrix as a flann::Matrix?
    //    eigen is column-major by default, flann is always row-major
    //    so, convert if necessary...

    // loop over each pair of bags
    //    compute the necessary rhos / nus
    //        construct indices if necessary
    //    call the div funcs and insert them in the matrices

    // think about:
    //    how should threads be split up?
    //    use plain threads (boost?) or TBB?
    //    precompute the necessary stuff, or do locking to compute as needed?

    std::vector<Eigen::MatrixXf> vec;
    return vec;
}

#endif
