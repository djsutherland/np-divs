#include <gtest/gtest.h>

#include "div_func.hpp"
#include "div_l2.hpp"
#include "div_bc.hpp"
#include "div_renyi.hpp"
#include "div_hellinger.hpp"
#include "dkn.hpp"
#include "np_divs.hpp"

#include <algorithm>

#include <boost/assign/std/vector.hpp>

#include <flann/flann.hpp>
#include <flann/io/hdf5.h>

using namespace boost::assign; // for vector +=
using namespace std;
using namespace flann;
using namespace NPDivs;

namespace {

class NPDivTest : public ::testing::Test {
    protected:

    NPDivTest() : index_params(AutotunedIndexParams(.99, 0, 0, 1)),
                  search_params(SearchParams(FLANN_CHECKS_AUTOTUNED)) {
        // setup work
    }

    virtual ~NPDivTest() {
        // cleanup work
    }

    // objects common to tests
    const IndexParams index_params;
    const SearchParams search_params;
};

// Tests that DKN works properly in a specific 2d case
TEST_F(NPDivTest, DKNTwoD) {
    float d[] = { -2.999, -5.672,
                  -9.051, -1.417,
                   2.066, -0.519,
                  -0.859, -8.354,
                   2.159, -0.470,
                  -5.365, -0.469,
                   9.829,  2.735,
                  -7.356, -9.513,
                  -2.687,  2.312,
                  -9.168, -2.966 };
    Matrix<float> dataset(d, 10, 2);

    float q[] = { -2.920, -9.522,
                   2.363,  6.885,
                   0.963,  4.673,
                   6.671,  0.481 };
    Matrix<float> query(q, 4, 2);

    vector<float> expected;
    expected += 3.8511, 7.3594, 5.2820, 4.6111;

    Index<L2<float> > index(dataset, index_params);
    index.buildIndex();

    vector<float> results = NPDivs::DKN(index, query, 2, search_params);

    for (size_t i = 0; i < expected.size(); i++)
        EXPECT_NEAR(results[i], expected[i], .01);
}


// Tests a bunch of divergence functions on data from test_dists.hdf5
TEST_F(NPDivTest, NPDivsGaussiansToSelf) {
    using flann::load_from_file;

    typedef flann::Matrix<float> Matrix;

    // load datasets
    const int num_groups = 2;
    const int num_per_group = 5;
    const int num_bags = num_groups * num_per_group;

    const string fname = "test_dists.hdf5";
    boost::format path("gaussian/%d/%d");

    Matrix* bags = new Matrix[num_bags];
    for (size_t group = 0; group < num_groups; group++) {
        for (size_t i = 0; i < num_per_group; i++) {
            load_from_file(
                    bags[group*num_per_group + i],
                    fname, (path % (group+1) % (i+1)).str());
        }
    }

    // specify divergence functions
    boost::ptr_vector<DivFunc> div_funcs;
    div_funcs.push_back(new DivL2());
    div_funcs.push_back(new DivRenyi(.999));
    div_funcs.push_back(new DivHellinger());
    div_funcs.push_back(new DivBC());

    const size_t num_df = div_funcs.size();

    // load expectations
    Matrix* expected = new Matrix[num_df];
    for (size_t i = 0; i < num_df; i++) {
        load_from_file(expected[i],
                fname, "gaussian/divs/" + div_funcs[i].name());
        ASSERT_EQ(expected[i].rows, num_bags);
        ASSERT_EQ(expected[i].cols, num_bags);
    }

    // preallocate results
    Matrix* results = new Matrix[num_df];
    for (size_t i = 0; i < num_df; i++) {
        results[i] = Matrix(new float[num_bags*num_bags], num_bags, num_bags);
    }

    // compute!
    np_divs(bags, num_bags, div_funcs, results, 3);

    // compare to expectations
    for (size_t df = 0; df < num_df; df++)
        for (size_t i = 0; i < num_bags; i++)
            for (size_t j = 0; j < num_bags; j++)
                EXPECT_NEAR(results[df][i][j], expected[df][i][j], .015) <<
                    boost::format("Big difference for df=%d, i=%d, j=%d")
                    % df % i % j;

    // deallocate datasets
    for (size_t i = 0; i < num_bags; i++)
        delete[] bags[i].ptr();
    delete[] bags;

    // deallocate expectations
    for (size_t i = 0; i < num_df; i++)
        delete[] expected[i].ptr();
    delete[] expected;

    // deallocate results
    for (size_t i = 0; i < num_df; i++)
        delete[] results[i].ptr();
    delete[] results;
}


}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
