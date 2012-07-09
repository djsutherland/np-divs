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
#ifndef NPDIVS_DIV_PARAMS_HPP_
#define NPDIVS_DIV_PARAMS_HPP_
#include "np-divs/basics.hpp"

#include <flann/flann.hpp>
#include <boost/function.hpp>

namespace npdivs {

void do_nothing(size_t);
void print_progress_cerr(size_t);

struct DivParams {
    int k; // the k of our k-nearest-neighbor searches
    flann::IndexParams index_params;
    flann::SearchParams search_params;
    size_t num_threads; // 0 means boost::thread::hardware_concurrency()

    size_t show_progress; // show progress every X steps; 0 means never
    boost::function<void (size_t)> print_progress;

    DivParams(
        int k = 3,
        flann::IndexParams index_params = flann::KDTreeSingleIndexParams(),
        flann::SearchParams search_params = flann::SearchParams(-1),
        size_t num_threads = 0,
        size_t show_progress = 1000,
        void (*print_progress)(size_t) = &print_progress_cerr)
    :
        k(k), index_params(index_params), search_params(search_params),
        num_threads(num_threads), show_progress(show_progress),
        print_progress(boost::function<void (size_t)>(
                print_progress == NULL ? &do_nothing : print_progress
        ))
    { }

    DivParams(int k,
            flann::IndexParams index_params, flann::SearchParams search_params,
            size_t num_threads, size_t show_progress,
            boost::function<void(size_t)> print_progress)
    :
        k(k), index_params(index_params), search_params(search_params),
        num_threads(num_threads), show_progress(show_progress),
        print_progress(print_progress)
    { }


};

flann::IndexParams index_params_from_str(const std::string &spec);

}
#endif
