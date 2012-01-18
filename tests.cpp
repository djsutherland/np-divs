#include <gtest/gtest.h>

#include "dkn.hpp"

#include <algorithm>
#include <boost/assign/std/vector.hpp>
#include <flann/flann.h>

using namespace boost::assign; // for vector +=
using namespace std;
using namespace flann;

namespace {

class DKNTest : public ::testing::Test {
    protected:

    DKNTest() : index_params(AutotunedIndexParams(.99, 0, 0, 1)),
                search_params(SearchParams(FLANN_CHECKS_AUTOTUNED)) {
        // setup work
    }

    virtual ~DKNTest() {
        // cleanup work
    }

    // objects common to tests
    const IndexParams index_params;
    const SearchParams search_params;
};

TEST_F(DKNTest, DKNTwoD) {
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
    vector<float> results = DKN(index, query, 2, search_params);

    for (size_t i = 0; i < expected.size(); i++)
        EXPECT_NEAR(results[i], expected[i], .01);
}

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
