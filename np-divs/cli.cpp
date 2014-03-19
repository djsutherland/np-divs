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
#include "np-divs/np_divs.hpp"
#include "np-divs/matrix_io.hpp"
#include "np-divs/div-funcs/div_func.hpp"
#include "np-divs/div-funcs/from_str.hpp"

#include <iostream>
#include <fstream>
#include <string>

#include <boost/bind.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/program_options.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/utility.hpp>

#include <flann/util/matrix.h>

using namespace npdivs;
using namespace std;
using namespace boost;
namespace po = boost::program_options;

struct ProgOpts : noncopyable {
    string x_bags_file;
    string y_bags_file;
    string results_file;

    ptr_vector<DivFunc> div_funcs;

    size_t k;
    size_t num_threads;

    flann::IndexParams index_params;
    flann::SearchParams search_params;

    size_t show_progress;

    void parse_div_funcs(const vector<string> &names) {
        for (size_t i = 0; i < names.size(); i++) {
            div_funcs.push_back(div_func_from_str(names[i]));
        }
    }

    void parse_index(const string name) {
        index_params = index_params_from_str(name);
    }
};


bool parse_args(int argc, char ** argv, ProgOpts& opts);

int main(int argc, char ** argv) {
    typedef flann::Matrix<double> Matrix;

    try {
        ProgOpts opts;
        opts.search_params = flann::SearchParams(64);
        if (!parse_args(argc, argv, opts))
            return 1;

        size_t num_df = opts.div_funcs.size();
        if (num_df == 0) {
            cerr << "Error: at least one div func is required\n";
            return 1;
        }

        // load input bags
        // TODO - gracefully handle nonexisting files
        size_t num_x;
        Matrix* x_bags;
        if (opts.x_bags_file == "-") {
            x_bags = matrices_from_csv(cin, num_x);
        } else {
            ifstream ifs(opts.x_bags_file.c_str(), ifstream::in);
            x_bags = matrices_from_csv(ifs, num_x);
            if (opts.show_progress)
                cerr << "Read " << num_x << " x bags.\n";
        }

        size_t num_y;
        Matrix* y_bags;
        if (opts.y_bags_file.empty()) {
            y_bags = NULL;
            num_y = num_x;
        } else if (opts.y_bags_file == "-") {
            y_bags = matrices_from_csv(cin, num_y);
        } else {
            ifstream ifs(opts.y_bags_file.c_str(), ifstream::in);
            y_bags = matrices_from_csv(ifs, num_y);
            if (opts.show_progress)
                cerr << "Read " << num_x << " y bags.\n";
        }


        Matrix* results = alloc_matrix_array<double>(num_df, num_x, num_y);

        DivParams params(opts.k, opts.index_params, opts.search_params,
                opts.num_threads, opts.show_progress);

        boost::posix_time::ptime t_start = boost::posix_time::second_clock::local_time();

        np_divs(x_bags, num_x, y_bags, num_y, opts.div_funcs, results, params);

        boost::posix_time::ptime t_end = boost::posix_time::second_clock::local_time();
        if (opts.show_progress)
            cerr << "Computation took " << (t_end - t_start).total_seconds() << " seconds.\n";

        if (opts.results_file == "-") {
            matrix_array_to_csv(cout, results, num_df);
        } else {
            ofstream ofs(opts.results_file.c_str());
            matrix_array_to_csv(ofs, results, num_df);
        }

        free_matrix_array(results, num_df);

    } catch (std::exception &e) {
        cerr << "Error: " << e.what() << endl;
        exit(1);
    }

    return 0;
}


// TODO nicer handling of matrix inputs
// TODO optionally support HDF5 inputs
// TODO positional arguments for x_bags, y_bags
// TODO support setting {index,search}_params
bool parse_args(int argc, char ** argv, ProgOpts& opts) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Produce this help message.")
        ("x-bags,x",
            po::value<string>(&opts.x_bags_file)->default_value("-"),
            "CSV-style file containing matrices separated by blank lines; "
            "- means stdin.")
        ("y-bags,y",
            po::value<string>(&opts.y_bags_file),
            "CSV-style file containing matrices separated by blank lines. "
            "If passed, divergences are calculated between the x bags (rows "
            "in the output matrix) and the y bags (columns); if not passed, "
            "divergences are calculated from the x bags to themselves. Pass "
            "- to use stdin; if both x and y are from stdin, x is read first "
            "and the two groups must be separated by exactly two blank lines. ")
        ("results,r",
            po::value<string>(&opts.results_file)->default_value("-"),
            "Where to write CSV-style matrix of results, where m[i,j] = "
            "div(x_i, y_j). Use - for stdout.")
        ("div-func,f",
            po::value< vector<string> >()->composing()
               ->notifier(bind(&ProgOpts::parse_div_funcs, boost::ref(opts), _1)),
            "Divergence functions to use; can be specified more than once. If "
            "none passed, uses L2. Format is name:arg1:arg2:..., where argN "
            "refers to the Nth argument to the corresponding DivFunc's "
            "constructor. Some support a first argument specifying a "
            "parameter: renyi:.99 means the Renyi-.99 divergence. All support "
            "a last parameter, which determines the way that large values are "
            "normalized: l2:.95 or renyi:.99:.95 means certain calculated "
            "intermediate values above the 95th percentile are cut down; 1 "
            "means not to do this; default is .99. All extra arguments are "
            "optional.")
        ("num-threads",
            po::value<size_t>(&opts.num_threads)->default_value(0),
            "Number of threads to use for calculations. 0 means one per core.")
        ("neighbors,k",
            po::value<size_t>(&opts.k)->default_value(3),
            "The k for k-nearest-neighbor calculations.")
        ("index,i",
            po::value<string>()->default_value("kdtree")
                ->notifier(bind(&ProgOpts::parse_index, boost::ref(opts), _1)),
            "The nearest-neighbor index to use. Options: linear, kdtree.")
        ("progress,p",
            po::value<size_t>(&opts.show_progress)->default_value(1000),
            "Show progress indications every X computations (default 1000; "
            "use 0 to not show any).")
    ;

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            exit(0);
        }

        po::notify(vm);

    } catch (std::exception &e) {
        cerr << "Error: " << e.what() << endl;
        return false;
    }

    return true;
}
