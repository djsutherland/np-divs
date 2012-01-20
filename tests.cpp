#include <gtest/gtest.h>

#include "div_func.hpp"
#include "div_l2.hpp"
#include "div_bc.hpp"
#include "div_renyi.hpp"
#include "div_hellinger.hpp"
#include "dkn.hpp"
#include "np_divs.hpp"

#include <algorithm>
#include <cassert>

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

void expect_near_matrix_array(
    const Matrix *results, const Matrix *expected, size_t num, float dist=.015)
{
    for (size_t n = 0; n < num; n++) {
        const Matrix &m = results[n];
        for (size_t i = 0; i < m.rows; i++)
            for (size_t j = 0; j < m.cols; j++)
                EXPECT_NEAR(results[n][i][j], expected[n][i][j], dist)
                    << boost::format("Big difference for n=%d, i=%d, j=%d")
                       % n % i % j;
    }
}

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
        assert (div_funcs.size() == num_df); // can't do ASSERT_EQ in ctor on gcc

        // load expectations
        for (size_t i = 0; i < num_df; i++) {
            load_from_file(expected[i],
                    fname, "gaussian/divs/" + div_funcs[i].name());
            assert (expected[i].rows == num_bags);
            assert (expected[i].cols == num_bags);
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

    Matrix* one_to_two_expected() {
        Matrix* real =
            alloc_matrix_array<float>(num_df, num_per_group, num_per_group);
        for (size_t df = 0; df < num_df; df++)
            for (size_t i = 0; i < num_per_group; i++)
                for (size_t j = 0; j < num_per_group; j++)
                    real[df][i][j] = expected[df][i][num_per_group + j];
        return real;
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


TEST_F(NPDivGaussiansTest, NPDivsGaussiansToSelfSingleThreaded) {
    Matrix* results = alloc_matrix_array<float>(num_df, num_bags, num_bags);

    np_divs(bags, num_bags, div_funcs, results, 3,
            index_params, search_params, 1);

    expect_near_matrix_array(results, expected, num_df);

    free_matrix_array(results, num_df);
}

TEST_F(NPDivGaussiansTest, NPDivsGaussiansToSelfTwoThreaded) {
    Matrix* results = alloc_matrix_array<float>(num_df, num_bags, num_bags);

    np_divs(bags, num_bags, div_funcs, results, 3,
            index_params, search_params, 2);

    expect_near_matrix_array(results, expected, num_df);

    free_matrix_array(results, num_df);
}

TEST_F(NPDivGaussiansTest, NPDivsGaussiansToSelfManyThreaded) {
    Matrix* results = alloc_matrix_array<float>(num_df, num_bags, num_bags);

    np_divs(bags, num_bags, div_funcs, results, 3,
            index_params, search_params, 50);

    expect_near_matrix_array(results, expected, num_df);

    free_matrix_array(results, num_df);
}



TEST_F(NPDivGaussiansTest, NPDivGaussiansOneToTwoSingleThreaded) {
    Matrix* results =
        alloc_matrix_array<float>(num_df, num_per_group, num_per_group);
    Matrix* real_expected = one_to_two_expected();

    np_divs(bags, num_per_group, bags + num_per_group, num_per_group,
            div_funcs, results, 3, index_params, search_params, 1);

    expect_near_matrix_array(results, real_expected, num_df);

    free_matrix_array(results, num_df);
    free_matrix_array(real_expected, num_df);
}

TEST_F(NPDivGaussiansTest, NPDivGaussiansOneToTwoTwoThreaded) {
    Matrix* results =
        alloc_matrix_array<float>(num_df, num_per_group, num_per_group);
    Matrix* real_expected = one_to_two_expected();

    np_divs(bags, num_per_group, bags + num_per_group, num_per_group,
            div_funcs, results, 3, index_params, search_params, 2);

    expect_near_matrix_array(results, real_expected, num_df);

    free_matrix_array(results, num_df);
    free_matrix_array(real_expected, num_df);
}

TEST_F(NPDivGaussiansTest, NPDivGaussiansOneToTwoManyThreaded) {
    Matrix* results =
        alloc_matrix_array<float>(num_df, num_per_group, num_per_group);
    Matrix* real_expected = one_to_two_expected();

    np_divs(bags, num_per_group, bags + num_per_group, num_per_group,
            div_funcs, results, 3, index_params, search_params, 50);

    expect_near_matrix_array(results, real_expected, num_df);

    free_matrix_array(results, num_df);
    free_matrix_array(real_expected, num_df);
}


} // end namespace


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
