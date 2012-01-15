#ifndef NP_DIVS_HPP_
#define NP_DIVS_HPP_

#include <vector>
#include <flann/flann.hpp>
#include <Eigen/Core>

#include "div_func.hpp"

template <typename Derived>
std::vector<Eigen::MatrixXf> np_divs(
        std::vector<Eigen::DenseBase<Derived> > bags,
        unsigned int k = 3);

template <typename Derived>
std::vector<Eigen::MatrixXf> np_divs(
        std::vector<Eigen::DenseBase<Derived> > bags,
        std::vector<DivFunc> div_funcs,
        unsigned int k = 3);

template <typename Derived>
std::vector<Eigen::MatrixXf> np_divs(
        std::vector<Eigen::DenseBase<Derived> > x_bags,
        std::vector<Eigen::DenseBase<Derived> > y_bags,
        unsigned int k = 3);

template <typename Derived>
std::vector<Eigen::MatrixXf> np_divs(
        std::vector<Eigen::DenseBase<Derived> > x_bags,
        std::vector<Eigen::DenseBase<Derived> > y_bags,
        std::vector<DivFunc> div_funcs,
        unsigned int k = 3);

#endif
