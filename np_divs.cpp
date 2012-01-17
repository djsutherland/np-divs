#include "np_divs.hpp"

#include <vector>
#include <cstdio>

#include <flann/flann.hpp>
#include <flann/io/hdf5.h>

#include <boost/format.hpp>

#include "div_hellinger.hpp"
#include "div_l2.hpp"
#include "div_renyi.hpp"

// most code in header because it's templated

using namespace std;

void test_np_divs(const string fname="test_dists.hdf5") {
    typedef flann::Matrix<float> Matrix;
    typedef vector<Matrix>::const_iterator c_iter;

    vector<Matrix> bags_std1;
    bags_std1.reserve(5);
    for (int i = 1; i <= 5; i++) {
        string path = (boost::format("gaussian1/%d") % i).str();

        Matrix dataset;
        flann::load_from_file(dataset, fname, path);

        bags_std1.push_back(dataset);
    }

    vector<Matrix> bags_std2;
    bags_std1.reserve(5);
    for (int i = 1; i <= 5; i++) {
        string path = (boost::format("gaussian2/%d") % i).str();

        Matrix dataset;
        flann::load_from_file(dataset, fname, path);

        bags_std2.push_back(dataset);
    }

    boost::ptr_vector<DivFunc> div_funcs;
    div_funcs.push_back(new DivL2());
    div_funcs.push_back(new DivRenyi(.999));
    div_funcs.push_back(new DivHellinger());

    vector<Matrix> x_bags;
    x_bags.insert(x_bags.end(), bags_std1.begin(), bags_std2.end());
    x_bags.insert(x_bags.end(), bags_std2.begin(), bags_std2.end());

    vector<Matrix> divs = np_divs(x_bags, div_funcs, 3);

    for (c_iter d = divs.begin(); d != divs.end(); d++) {
        for (size_t i = 0; i < d->rows; i++) {
            for (size_t j = 0; j < d-> cols; j++) {
                cout << (*d)[i][j] << " ";
            }
            cout << endl;
        }
        cout << endl << endl;
    }

    for (c_iter i = bags_std1.begin(); i != bags_std1.end(); i++)
        delete[] i->ptr();
    for (c_iter i = bags_std2.begin(); i != bags_std2.end(); i++)
        delete[] i->ptr();
}

int main() {
    test_np_divs();
    return 0;
}
