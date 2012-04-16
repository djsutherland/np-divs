/*******************************************************************************
 * Copyright (c) 2012, Dougal J. Sutherland (dsutherl@cs.cmu.edu).             *
 * All rights reserved.                                                        *
 *                                                                             *
 * Redistribution and use in source and binary forms, with or without          *
 * modification, are permitted provided that the following conditions are met: *
 *                                                                             *
 *     * Redistributions of source code must retain the above copyright        *
 *       notice, this list of conditions and the following disclaimer.         *
 *                                                                             *
 *     * Redistributions in binary form must reproduce the above copyright     *
 *       notice, this list of conditions and the following disclaimer in the   *
 *       documentation and/or other materials provided with the distribution.  *
 *                                                                             *
 *     * Neither the name of Carnegie Mellon University nor the                *
 *       names of the contributors may be used to endorse or promote products  *
 *       derived from this software without specific prior written permission. *
 *                                                                             *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" *
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE   *
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE  *
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE   *
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR         *
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF        *
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS    *
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN     *
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)     *
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE  *
 * POSSIBILITY OF SUCH DAMAGE.                                                 *
 ******************************************************************************/
#include "np-divs/basics.hpp"
#include <gtest/gtest.h>

#include "np-divs/div-funcs/div_func.hpp"
#include "np-divs/div-funcs/div_l2.hpp"
#include "np-divs/div-funcs/div_bc.hpp"
#include "np-divs/div-funcs/div_renyi.hpp"
#include "np-divs/div-funcs/div_hellinger.hpp"
#include "np-divs/dkn.hpp"
#include "np-divs/fix_terms.hpp"
#include "np-divs/gamma.hpp"
#include "np-divs/np_divs.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <string>

#include <boost/assign/std/vector.hpp>

#include <flann/flann.hpp>
#include <flann/io/hdf5.h>

using namespace boost::assign; // for vector +=
using namespace std;
using namespace npdivs;

using flann::load_from_file;
using flann::Index;
using flann::IndexParams;
using flann::SearchParams;
using flann::L2;

typedef flann::Matrix<float> MatrixF;
typedef flann::Matrix<double> MatrixD;

void expect_near_matrix_array(
    const MatrixD *results, const MatrixD *expected,
    size_t num, float dist=.001)
{
    float error;
    for (size_t n = 0; n < num; n++) {
        const MatrixD &m = results[n];
        for (size_t i = 0; i < m.rows; i++) {
            for (size_t j = 0; j < m.cols; j++) {
                error = max(expected[n][i][j] * dist, 1e-5);
                EXPECT_NEAR(results[n][i][j], expected[n][i][j], error)
                    << boost::format("Big difference for n=%d, i=%d, j=%d")
                       % n % i % j;
            }
        }
    }
}

namespace {


void test_fix_terms(vector<float> &terms, vector<float> &expected, double ub) {
    std::sort(expected.begin(), expected.end());

    fix_terms(terms, ub);
    std::sort(terms.begin(), terms.end());

    for (size_t i = 0; i < terms.size(); i++)
        if (!isinf(terms[i]) || !isinf(expected[i]))
            EXPECT_NEAR(terms[i], expected[i], 1e-4);
}

TEST(UtilitiesTest, FixTermsSimple100) {
    vector<float> terms;
    terms += 0.5377, 1.8339, -2.2588, 0.8622, 0.3188, -1.3077, -0.4336, 0.3426,
          3.5784, 2.7694, -1.3499, 3.0349, 0.7254, -0.0631, 0.7147, -0.2050,
          -0.1241, 1.4897, 1.4090, 1.4172, 0.6715, -1.2075, 0.7172, 1.6302,
          0.4889, 1.0347, 0.7269, -0.3034, 0.2939, -0.7873, 0.8884, -1.1471,
          -1.0689, -0.8095, -2.9443, 1.4384, 0.3252, -0.7549, 1.3703, -1.7115,
          -0.1022, -0.2414, 0.3192, 0.3129, -0.8649, -0.0301, -0.1649, 0.6277,
          1.0933, 1.1093, -0.8637, 0.0774, -1.2141, -1.1135, -0.0068, 1.5326,
          -0.7697, 0.3714, -0.2256, 1.1174, -1.0891, 0.0326, 0.5525, 1.1006,
          1.5442, 0.0859, -1.4916, -0.7423, -1.0616, 2.3505, -0.6156, 0.7481,
          -0.1924, 0.8886, -0.7648, -1.4023, -1.4224, 0.4882, -0.1774, -0.1961,
          1.4193, 0.2916, 0.1978, 1.5877, -0.8045, 0.6966, 0.8351, -0.2437,
          0.2157, -1.1658, -1.1480, 0.1049, 0.7223, 2.5855, -0.6669, 0.1873,
          -0.0825, -1.9330, -0.4390, -1.7947;

    vector<float> expected = terms;
    expected[8] = 3.3067;

    test_fix_terms(terms, expected, .99);
}

TEST(UtilitiesTest, FixTermsOther100) {
    vector<float> terms;
    terms += 0.9195, -0.0617, -1.3385, -0.1218, -2.1000, 1.8291, 0.0549,
          -0.0789, -0.2861, 0.3739, 0.0027, -1.2186, -1.2901, 1.1663, -0.8901,
          1.4472, -1.7756, -0.8204, -1.0579, 1.0077, -0.4595, 0.7860, -0.8349,
          0.6164, -0.4736, 0.1797, 0.6522, 6.2691, 8.1633, 1.1778, -0.9921,
          -0.7535, 1.4361, 0.3297, -0.5314, 1.7876, 0.0150, -0.7715, -0.8813,
          1.1515, 0.6752, 0.3413, -1.1232, 0.6571, 3.2662, 0.2452, -0.1967,
          -0.0537, 1.2281, -0.1495, -0.8551, 0.2521, 0.9336, 2.1212, 0.6745,
          0.1170, 0.8917, -0.0123, -2.1032, -2.1327, 1.4584, 0.8522, -0.8366,
          0.9018, 1.3986, 0.3386, -0.2276, -0.7302, -0.9163, 0.0853, 1.2486,
          0.0560, 0.9663, 0.9855, 1.0368, 0.0317, 0.9394, -1.7035, -0.3171,
          -2.2082, 0.0728, 1.2559, -0.0835, 0.3500, -0.0683, -0.6434, 0.9107,
          -0.8301, 0.4882, -0.4319, -0.5635, -1.0781, 0.5531, 0.7233, 1.2353,
          0.1558, -0.6426, -0.5250, 0.2199, 0.2584;

    vector<float> expected = terms;
    expected[27] = expected[28] = expected[44] = 2.1212;

    test_fix_terms(terms, expected, .965);
}

TEST(UtilitiesTest, FixTermsWithInfAbove) {
    vector<float> terms;

    float inf = std::numeric_limits<float>().infinity();

    terms += 1.1802, -0.5111, -1.3504, -0.3443, -0.7929, -0.7879, 0.8764,
          19.6827, 0.2975, -0.1433, 16.8614, 16.2429, 0.7989, -0.2036, -0.5767,
          -0.8718, 0.1641, 0.0836, -1.2879, 0.1785, 0.6520, -1.2273, 1.3920,
          -1.1537, 1.1435, inf, 0.7007, 0.1004, 19.5552, 1.4390, -0.5372,
          0.1011, 0.3774, 0.0080, 0.1638, -0.0506, -0.5877, 1.1004, 0.9916,
          0.6633, 0.7530, -0.3251, 0.2590, 0.7998, -1.6068, inf, 0.6035, inf,
          -1.0864, 0.1909, -1.4197, 0.6826, 1.6760, 0.0179, -0.5544, 0.9308,
          2.5318, -0.2052, -0.7302, 0.5996, 0.2461, 0.3067, -0.2012, 2.0541,
          -0.7348, -0.4079, -1.0718, 1.4942, 0.6476, -0.2289, -1.2232, inf,
          -0.0615, 0.7256, -1.0711, -1.9654, -0.5362, 1.7854, 0.7884, 0.3270,
          -0.2929;

    vector<float> expected = terms;
    expected[7] = expected[10] = expected[11] = expected[25] = expected[28] =
        expected[45] = expected[47] = expected[71] = 8.0163;

    test_fix_terms(terms, expected, .9);
}


TEST(UtilitiesTest, FixTermsWithInfBelow) {
    vector<float> terms;

    float inf = std::numeric_limits<float>().infinity();

    terms += 1.1802, -0.5111, -1.3504, -0.3443, -0.7929, -0.7879, 0.8764,
          19.6827, 0.2975, -0.1433, 16.8614, 16.2429, 0.7989, -0.2036, -0.5767,
          -0.8718, 0.1641, 0.0836, -1.2879, 0.1785, 0.6520, -1.2273, 1.3920,
          -1.1537, 1.1435, inf, 0.7007, 0.1004, 19.5552, 1.4390, -0.5372,
          0.1011, 0.3774, 0.0080, 0.1638, -0.0506, -0.5877, 1.1004, 0.9916,
          0.6633, 0.7530, -0.3251, 0.2590, 0.7998, -1.6068, inf, 0.6035, inf,
          -1.0864, 0.1909, -1.4197, 0.6826, 1.6760, 0.0179, -0.5544, 0.9308,
          2.5318, -0.2052, -0.7302, 0.5996, 0.2461, 0.3067, -0.2012, 2.0541,
          -0.7348, -0.4079, -1.0718, 1.4942, 0.6476, -0.2289, -1.2232, inf,
          -0.0615, 0.7256, -1.0711, -1.9654, -0.5362, 1.7854, 0.7884, 0.3270,
          -0.2929;

    vector<float> expected = terms;
    expected[25] = expected[45] = expected[47] = expected[71] = 19.6827;

    test_fix_terms(terms, expected, .98);
}

TEST(UtilitiesTest, FixTermsWithInfAndNan) {
    vector<float> terms;

    float inf = std::numeric_limits<float>().infinity();
    float nan = std::numeric_limits<float>().quiet_NaN();

    terms += 0.2346, nan, 0.0160, nan, 1.1949, -1.4867, -0.0240, 3.7520,
          -0.9096, -0.5122, 0.1069, -0.3973, 0.7500, -1.3019, -0.9338, -0.2939,
          1.2118, -1.0767, -1.3027, 0.0099, 0.4957, -0.6932, -0.5446, -0.1583,
          0.4763, 1.0468, -0.0382, 0.5777, -0.4535, 1.1198, 0.8421, -0.4130,
          1.1107, 0.0005, 1.2196, 0.1620, 1.1247, -1.9055, -0.5186, -0.8005,
          0.6188, 0.8332, 0.8700, 0.9239, 9.8975, -0.2494, 0.2930, 1.7697, inf,
          -0.8324, 1.4550, -0.9705, -0.9090, -0.7298, -0.3125, 0.5379, 1.0355,
          1.1462, 0.2040, -1.1386, -0.0775, -0.9242, 2.4677, 6.2213, -1.1986,
          -0.0884, 0.5466, 0.6762, 0.2894, 2.2231, inf;

    vector<float> expected = terms;
    expected.erase(expected.begin() + 3);
    expected.erase(expected.begin() + 1);
    expected[46] = expected[68] = 9.8975;

    test_fix_terms(terms, expected, .98);
}

TEST(UtilitiesTest, Gamma) {
    using npdivs::gamma;

    // integers
    EXPECT_NEAR(gamma(1), 1, 1e-10);
    EXPECT_NEAR(gamma(2), 1, 1e-10);
    EXPECT_NEAR(gamma(3), 2, 1e-10);
    EXPECT_NEAR(gamma(4), 6, 1e-10);
    EXPECT_NEAR(gamma(5), 24, 1e-10);
    EXPECT_NEAR(gamma(13), 479001600, 1e-10);

    // half-integers
    EXPECT_NEAR(gamma(.5), sqrt(M_PI), 1e-10);
    EXPECT_NEAR(gamma(1.5), .886226925452758, 5e-16);
    EXPECT_NEAR(gamma(2.5), 1.329340388179137, 5e-16);
    EXPECT_NEAR(gamma(3.5), 3.323350970447843, 5e-16);
    EXPECT_NEAR(gamma(4.5), 11.631728396567450, 5e-16);
    EXPECT_NEAR(gamma(13.5), 1710542068.319572, 5e-6);

    // others
    EXPECT_NEAR(gamma(1.12345678), 0.942309030392057, 5e-16);
    EXPECT_NEAR(gamma(7.1525), 959.701709437015, 5e-13);
}

TEST(UtilitiesTest, LogGamma) {
    using npdivs::lgamma;

    // integers
    EXPECT_NEAR(lgamma(1), 0, 1e-15);
    EXPECT_NEAR(lgamma(2), 0, 1e-15);
    EXPECT_NEAR(lgamma(3), log(2), 1e-15);
    EXPECT_NEAR(lgamma(4), log(6), 1e-15);
    EXPECT_NEAR(lgamma(5), log(24), 1e-15);
    EXPECT_NEAR(lgamma(13), log(479001600), 1e-10);
    EXPECT_NEAR(lgamma(10000), 82099.71749644238, 1e-10);

    // half-integers
    EXPECT_NEAR(lgamma(.5), log(M_PI) / 2, 1e-10);
    EXPECT_NEAR(lgamma(1.5), -0.120782237635245, 5e-15);
    EXPECT_NEAR(lgamma(2.5), 0.284682870472919, 5e-15);
    EXPECT_NEAR(lgamma(3.5), 1.200973602347074, 5e-15);
    EXPECT_NEAR(lgamma(4.5), 2.453736570842442, 5e-15);
    EXPECT_NEAR(lgamma(13.5), 21.260076156244700, 5e-6);
    EXPECT_NEAR(lgamma(852.5), 4897.862610487247, 1e-12);

    // others
    EXPECT_NEAR(lgamma(1.12345678), -0.059422000463912, 5e-16);
    EXPECT_NEAR(lgamma(7.1525), 6.866622516842186, 5e-13);
    EXPECT_NEAR(lgamma(62314.156), 625626.0295132722, 5e-8);
}


class NPDivTest : public ::testing::Test {
    protected:

    NPDivTest() :
        params(DivParams(
                    3, // k
                    flann::KDTreeSingleIndexParams(),
                    SearchParams(flann::FLANN_CHECKS_UNLIMITED),
                    0, // threads
                    0)) // print progress every
    {}

    virtual ~NPDivTest() {}

    DivParams params;
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
    MatrixF dataset(d, 10, 2);

    float q[] = { -2.920, -9.522,
                   2.363,  6.885,
                   0.963,  4.673,
                   6.671,  0.481 };
    MatrixF query(q, 4, 2);

    vector<float> expected;
    expected += 3.8511, 7.3594, 5.2820, 4.6111;

    Index<L2<float> > index(dataset, params.index_params);
    index.buildIndex();

    vector<float> results = npdivs::DKN(index, query, 2, params.search_params);

    for (size_t i = 0; i < expected.size(); i++)
        EXPECT_NEAR(results[i], expected[i], .01);
}


class NPDivDataTest : public NPDivTest {
    typedef NPDivTest super;

    protected:
    NPDivDataTest() :
        fname("test_dists.hdf5"),
        num_groups(2),
        num_per_group(5),
        num_bags(num_groups * num_per_group),
        bags(new MatrixF[num_bags]),

        num_df(4),
        expected(new MatrixD[num_df])
    {
        // specify divergence functions
        div_funcs.push_back(new DivL2());
        div_funcs.push_back(new DivRenyi(.999));
        div_funcs.push_back(new DivHellinger());
        div_funcs.push_back(new DivBC());
        assert (div_funcs.size() == num_df);
        // ^^ can't do ASSERT_EQ in constructor on old gcc
    }

    void load_bags(string groupname) {
        boost::format f = boost::format("%s/%i/%i");

        // load bags
        for (size_t group = 0; group < num_groups; group++) {
            for (size_t i = 0; i < num_per_group; i++) {
                load_from_file(
                        bags[group*num_per_group + i],
                        fname, (f % groupname % (group+1) % (i+1)).str());
            }
        }

        // load expectations
        for (size_t i = 0; i < num_df; i++) {
            load_from_file(expected[i],
                    fname, groupname + "/divs/" + div_funcs[i].name());
            assert (expected[i].rows == num_bags);
            assert (expected[i].cols == num_bags);
        }
    }

    void free_bags() {
        // doesn't free the bags/expected array itself; deconstructor does that
        for (size_t i = 0; i < num_bags; i++)
            delete[] bags[i].ptr();

        for (size_t i = 0; i < num_df; i++)
            delete[] expected[i].ptr();
    }

    void test_to_self(size_t num_threads = 0) {
        MatrixD* results =
            alloc_matrix_array<double>(num_df, num_bags, num_bags);

        params.num_threads = num_threads;
        np_divs(bags, num_bags, div_funcs, results, params);

        expect_near_matrix_array(results, expected, num_df);

        free_matrix_array(results, num_df);
    }

    void test_one_to_two(size_t num_threads = 0) {
        MatrixD* results =
            alloc_matrix_array<double>(num_df, num_per_group, num_per_group);

        MatrixD* _expected =
            alloc_matrix_array<double>(num_df, num_per_group, num_per_group);
        for (size_t df = 0; df < num_df; df++)
            for (size_t i = 0; i < num_per_group; i++)
                for (size_t j = 0; j < num_per_group; j++)
                    _expected[df][i][j] = expected[df][i][num_per_group + j];

        params.num_threads = num_threads;
        np_divs(bags, num_per_group, bags + num_per_group, num_per_group,
                div_funcs, results, params);

        expect_near_matrix_array(results, _expected, num_df);

        free_matrix_array(results, num_df);
        free_matrix_array(_expected, num_df);
    }

    const string fname;
    const size_t num_groups;
    const size_t num_per_group;
    const size_t num_bags;
    MatrixF* bags;

    boost::ptr_vector<DivFunc> div_funcs;
    const size_t num_df;
    MatrixD* expected;
};


class Gaussians2DTest : public NPDivDataTest {
    typedef NPDivDataTest super;
protected:
    Gaussians2DTest() : super() { load_bags("gaussian"); }
    ~Gaussians2DTest() { free_bags(); }
};

TEST_F(Gaussians2DTest, ToSelfOneThread)   { test_to_self(1); }
TEST_F(Gaussians2DTest, ToSelfTwoThreads)  { test_to_self(2); }
TEST_F(Gaussians2DTest, ToSelfManyThreads) { test_to_self(50); }

TEST_F(Gaussians2DTest, OneToTwoOneThread)   { test_one_to_two(1); }
TEST_F(Gaussians2DTest, OneToTwoTwoThreads)  { test_one_to_two(2); }
TEST_F(Gaussians2DTest, OneToTwoManyThreads) { test_one_to_two(50); }


class Gaussians50DTest : public NPDivDataTest {
    typedef NPDivDataTest super;
protected:
    Gaussians50DTest() : super() {
        params.index_params = flann::LinearIndexParams();
        load_bags("gaussian-50");
    }
    ~Gaussians50DTest() { free_bags(); }
};

TEST_F(Gaussians50DTest, DISABLED_ToSelf) { test_to_self(); }

TEST_F(Gaussians50DTest, DISABLED_OneToTwo) { test_one_to_two(); }


} // end namespace


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
