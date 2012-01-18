#include "np_divs.hpp"

#include <vector>
#include <cstdio>

#include <flann/flann.hpp>
#include <flann/io/hdf5.h>

#include <boost/format.hpp>

#include "div_bc.hpp"
#include "div_hellinger.hpp"
#include "div_l2.hpp"
#include "div_renyi.hpp"

// most code in header because it's templated

using namespace std;

void test_np_divs(const string fname="test_dists.hdf5") {
    typedef flann::Matrix<float> Matrix;

    // load bags
#define NUM_STD1 5
    Matrix* bags_std1 = new Matrix[NUM_STD1];
    for (size_t i = 0; i < NUM_STD1; i++) {
        string path = (boost::format("gaussian1/%d") % (i+1)).str();
        flann::load_from_file(bags_std1[i], fname, path);
    }

#define NUM_STD2 5
    Matrix* bags_std2 = new Matrix[NUM_STD2];
    for (size_t i = 0; i < NUM_STD2; i++) {
        string path = (boost::format("gaussian2/%d") % (i+1)).str();
        flann::load_from_file(bags_std2[i], fname, path);
    }

    // combine bags into one array
    Matrix* x_bags = new Matrix[NUM_STD1 + NUM_STD2];
    for (size_t i = 0; i < NUM_STD1; i++)
        x_bags[i] = bags_std1[i];
    for (size_t i = 0; i < NUM_STD2; i++)
        x_bags[i+NUM_STD2] = bags_std2[i];
    size_t num_bags = NUM_STD1 + NUM_STD2;

    // specify divergence functions
    boost::ptr_vector<DivFunc> div_funcs;
    div_funcs.push_back(new DivL2());
    div_funcs.push_back(new DivRenyi(.999));
    div_funcs.push_back(new DivHellinger());
    div_funcs.push_back(new DivBC());

    size_t num_df = div_funcs.size();

    // preallocate results
    Matrix* results = new Matrix[num_df];
    for (size_t i = 0; i < num_df; i++) {
        results[i] = Matrix(new float[num_bags*num_bags], num_bags, num_bags);
    }

    // compute!
    np_divs(x_bags, NUM_STD1 + NUM_STD2, div_funcs, results, 3);

    // print out results
    for (size_t d = 0; d < num_df; d++) {
        Matrix m = results[d];
        cout << endl;
        for (size_t i = 0; i < m.rows; i++) {
            for (size_t j = 0; j < m.cols; j++) {
                cout << boost::format("%.3f  ") % m[i][j];
            }
            cout << endl;
        }
        cout << endl;
    }

    // deallocate datesets
    for (size_t i = 0; i < NUM_STD1; i++)
        delete bags_std1[i].ptr();
    for (size_t i = 0; i < NUM_STD2; i++)
        delete bags_std2[i].ptr();
    delete[] results;
}

int main() {
    test_np_divs();
    return 0;
}
