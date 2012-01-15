#include <Eigen/Core>
#include <Eigen/StdVector>
// TODO - be careful in handling std::vector<Eigen::whatever>
//        http://eigen.tuxfamily.org/dox/TopicStlContainers.html

#include "np_divs.hpp"
#include "div_l2.hpp"

using std::vector;

template <typename Derived>
Eigen::MatrixXf np_divs(
        vector<Eigen::DenseBase<Derived> > bags,
        unsigned int k)
{
    vector<DivFunc> div_funcs;
    div_funcs.push_back(DivL2());

    return np_divs(bags, div_funcs, k);
}

template <typename Derived>
Eigen::MatrixXf np_divs(
        vector<Eigen::DenseBase<Derived> > bags,
        vector<DivFunc> div_funcs,
        unsigned int k)
{
    // TODO - do this faster (ie not repeating work)
    return np_divs(bags, bags, div_funcs, k);
}

template <typename Derived>
Eigen::MatrixXf np_divs(
        vector<Eigen::DenseBase<Derived> > x_bags,
        vector<Eigen::DenseBase<Derived> > y_bags,
        unsigned int k)
{
    vector<DivFunc> div_funcs;
    div_funcs.push_back(DivL2());

    return np_divs(x_bags, y_bags, div_funcs, k);
}

template <typename Derived>
vector<Eigen::MatrixXf> np_divs(
        vector<Eigen::DenseBase<Derived> > x_bags,
        vector<Eigen::DenseBase<Derived> > y_bags,
        vector<DivFunc> div_funcs,
        unsigned int k)
{
    /* Calculates the matrix of divergences between x_bags and y_bags for
     * each of the passed div_funcs. Returns a vector of matrices corresponding
     * to the passed div_funcs. Rows of each matrix are an x_bag, columns are
     * a y_bag.
     */

    // is there a nice way to use an Eigen::Matrix as a flann::Matrix?
    //    eigen is column-major by default...what about flann?

    // loop over each pair of bags
    //    compute the necessary rhos / nus
    //        construct indices if necessary
    //    call the div funcs and insert them in the matrices

    // think about:
    //    how should threads be split up?
    //    use plain threads (boost?) or TBB?
    //    precompute the necessary stuff, or do some kind of locking to compute as needed?

    vector<Eigen::MatrixXf> vec;
    return vec;
}

int main() {
    return 1;
}
