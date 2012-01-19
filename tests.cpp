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
using namespace NPDivs;

using flann::load_from_file;
using flann::Index;
using flann::IndexParams;
using flann::SearchParams;
using flann::L2;

typedef flann::Matrix<float> Matrix;

namespace {

class NPDivTest : public ::testing::Test {
    protected:

    NPDivTest() : index_params(flann::KDTreeSingleIndexParams()),
                  search_params(SearchParams())
    {}

    virtual ~NPDivTest() {}

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
    Matrix dataset(d, 10, 2);

    float q[] = { -2.920, -9.522,
                   2.363,  6.885,
                   0.963,  4.673,
                   6.671,  0.481 };
    Matrix query(q, 4, 2);

    vector<float> expected;
    expected += 3.8511, 7.3594, 5.2820, 4.6111;

    Index<L2<float> > index(dataset, index_params);
    index.buildIndex();

    vector<float> results = NPDivs::DKN(index, query, 2, search_params);

    for (size_t i = 0; i < expected.size(); i++)
        EXPECT_NEAR(results[i], expected[i], .01);
}


class NPDivGaussiansTest : public NPDivTest {
    typedef NPDivTest super;

    protected:
    NPDivGaussiansTest() :
        fname("test_dists.hdf5"),
        num_groups(2),
        num_per_group(5),
        num_bags(num_groups * num_per_group),
        bags(new Matrix[num_bags]),

        num_df(4),
        expected(new Matrix[num_df])
    {
        // load bags
        boost::format path("gaussian/%d/%d");
        for (size_t group = 0; group < num_groups; group++) {
            for (size_t i = 0; i < num_per_group; i++) {
                load_from_file(
                        bags[group*num_per_group + i],
                        fname, (path % (group+1) % (i+1)).str());
            }
        }

        // specify divergence functions
        div_funcs.push_back(new DivL2());
        div_funcs.push_back(new DivRenyi(.999));
        div_funcs.push_back(new DivHellinger());
        div_funcs.push_back(new DivBC());
        ASSERT_EQ(div_funcs.size(), num_df);

        // load expectations
        for (size_t i = 0; i < num_df; i++) {
            load_from_file(expected[i],
                    fname, "gaussian/divs/" + div_funcs[i].name());
            ASSERT_EQ(expected[i].rows, num_bags);
            ASSERT_EQ(expected[i].cols, num_bags);
        }
    }

    virtual ~NPDivGaussiansTest() {
        // free bags
        for (size_t i = 0; i < num_bags; i++)
            delete[] bags[i].ptr();
        delete[] bags;

        // free expected
        for (size_t i = 0; i < num_df; i++)
            delete[] expected[i].ptr();
        delete[] expected;
    }

    const string fname;
    const size_t num_groups;
    const size_t num_per_group;
    const size_t num_bags;
    Matrix* bags;

    boost::ptr_vector<DivFunc> div_funcs;
    const size_t num_df;
    Matrix* expected;
};

// Tests a bunch of divergence functions on data from test_dists.hdf5
TEST_F(NPDivGaussiansTest, NPDivsGaussiansToSelf) {
    Matrix* results = alloc_matrix_array<float>(num_df, num_bags, num_bags);

    np_divs(bags, num_bags, div_funcs, results, 3, index_params, search_params);

    // compare to expectations
    for (size_t df = 0; df < num_df; df++)
        for (size_t i = 0; i < num_bags; i++)
            for (size_t j = 0; j < num_bags; j++)
                EXPECT_NEAR(results[df][i][j], expected[df][i][j], .015)
                    << boost::format("Big difference for df=%d, i=%d, j=%d")
                       % df % i % j;

    free_matrix_array(results, num_df);
}

TEST_F(NPDivGaussiansTest, NPDivGaussiansOneToTwo) {
    // copy out the upper-right block of expected, which is what we really want
    Matrix* real_expected = new Matrix[num_df];
    for (size_t df = 0; df < num_df; df++) {
        real_expected[df] = Matrix(new float[num_per_group*num_per_group],
                                   num_per_group, num_per_group);
        for (size_t i = 0; i < num_per_group; i++) {
            for (size_t j = 0; j < num_per_group; j++) {
                real_expected[df][i][j] = expected[df][i][num_per_group + j];
            }
        }
    }

    Matrix* results =
        alloc_matrix_array<float>(num_df, num_per_group, num_per_group);

    np_divs(bags, num_per_group, bags + num_per_group, num_per_group,
            div_funcs, results, 3, index_params, search_params);

    // // compare to expectations
    // for (size_t df = 0; df < num_df; df++)
    //     for (size_t i = 0; i < num_bags; i++)
    //         for (size_t j = 0; j < num_bags; j++)
    //             EXPECT_NEAR(results[df][i][j], real_expected[df][i][j], .015)
    //                 << boost::format("Big difference for df=%d, i=%d, j=%d")
    //                    % df % i % j;

    free_matrix_array(results, num_df);
}

} // end namespace


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
