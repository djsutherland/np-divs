#ifndef FIX_TERMS_HPP_
#define FIX_TERMS_HPP_

#include <Eigen/Dense>
#include <vector>

template <typename Derived>
std::vector<typename Derived::Scalar> as_vector(const Eigen::DenseBase<Derived> &v);

template <typename T>
void fix_terms(std::vector<T> &terms, double ub = .99);

#endif
