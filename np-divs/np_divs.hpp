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
#ifndef NPDIVS_NP_DIVS_HPP_
#define NPDIVS_NP_DIVS_HPP_
#include "np-divs/basics.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <boost/format.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/thread.hpp>
#include <boost/utility.hpp>
#include <boost/version.hpp>

#include <flann/flann.hpp>

#include "np-divs/div-funcs/div_func.hpp"
#include "np-divs/div-funcs/div_l2.hpp"
#include "np-divs/div_params.hpp"
#include "np-divs/dkn.hpp"
#include "np-divs/matrix_arrays.hpp"

namespace NPDivs {

////////////////////////////////////////////////////////////////////////////////
// Declarations of the main np_divs functions

#define INDEX_PARAMS flann::KDTreeSingleIndexParams()
#define SEARCH_PARAMS flann::SearchParams(64)
// TODO - figure out how to tune flann indices (better than autotuning each)

#define DEFAULT_DIV_FUNCS \
  boost::ptr_vector<DivFunc> div_funcs; \
  div_funcs.push_back(new DivL2());

// TODO: overloads that write into a vector< vector< vector<float> > >

template <typename Scalar>
void np_divs(
    const flann::Matrix<Scalar> *bags, size_t num_bags,
    flann::Matrix<double> *results,
    const DivParams &div_params,
    bool verify_results_alloced = true);

template <typename Scalar>
void np_divs(
    const flann::Matrix<Scalar> *bags, size_t num_bags,
    const DivFunc &div_func,
    flann::Matrix<double> *results,
    const DivParams &div_params,
    bool verify_results_alloced = true);

template <typename Scalar>
void np_divs(
    const flann::Matrix<Scalar> *bags, size_t num_bags,
    const boost::ptr_vector<DivFunc> &div_funcs,
    flann::Matrix<double> *results,
    const DivParams &div_params,
    bool verify_results_alloced = true);

template <typename Scalar>
void np_divs(
    const flann::Matrix<Scalar> *x_bags, size_t num_x,
    const flann::Matrix<Scalar> *y_bags, size_t num_y,
    flann::Matrix<double>* results,
    const DivParams &div_params,
    bool verify_results_alloced = true);

template <typename Scalar>
void np_divs(
    const flann::Matrix<Scalar> *x_bags, size_t num_x,
    const flann::Matrix<Scalar> *y_bags, size_t num_y,
    const DivFunc &div_func,
    flann::Matrix<double>* results,
    const DivParams &div_params,
    bool verify_results_alloced = true);

template <typename Scalar>
void np_divs(
    const flann::Matrix<Scalar> *x_bags, size_t num_x,
    const flann::Matrix<Scalar> *y_bags, size_t num_y,
    const boost::ptr_vector<DivFunc> &div_funcs,
    flann::Matrix<double>* results,
    const DivParams &div_params,
    bool verify_results_alloced = true);



////////////////////////////////////////////////////////////////////////////////
// Declarations of helpers used in the code below

inline size_t get_num_threads(size_t num_threads);

template <typename T>
void verify_allocated(
        flann::Matrix<T> *matrices,
        size_t num_matrices, size_t rows, size_t cols);
// throws a std::length_error if they're not the right size

template <typename Distance>
flann::Index<Distance>** make_indices(
        const flann::Matrix<typename Distance::ElementType> *datasets,
        size_t n,
        const flann::IndexParams index_params);

template <typename Distance>
inline void free_indices(flann::Index<Distance>** indices, size_t n);


template <typename Distance>
std::vector<std::vector<float> > get_rhos(
        const flann::Matrix<typename Distance::ElementType> *bags,
        flann::Index<Distance> **indices,
        size_t n,
        int k,
        const flann::SearchParams &search_params = SEARCH_PARAMS,
        size_t num_threads=1);


////////////////////////////////////////////////////////////////////////////////
// Functor classes used to do the computation work

template <typename Distance>
class divcalc_worker : boost::noncopyable {
    protected:

    typedef flann::Matrix<typename Distance::ElementType> Matrix;
    typedef flann::Index<Distance> Index;
    typedef std::vector<float> DistVec;
    typedef std::vector<DistVec> DistVecVec;
    typedef std::pair<size_t, size_t> size_pair;

    int k;
    int dim;

    const boost::ptr_vector<DivFunc> &div_funcs;
    size_t num_dfs;

    const flann::SearchParams &search_params;

    flann::Matrix<double> *results;
    boost::mutex &jobs_mutex;
    std::queue<size_pair> &jobs;

    public:

    divcalc_worker(
            int k,
            int dim,
            const boost::ptr_vector<DivFunc> &div_funcs,
            const flann::SearchParams &search_params,
            flann::Matrix<double> *results,
            boost::mutex &jobs_mutex,
            std::queue<size_pair> &jobs)
        :
            k(k), dim(dim), div_funcs(div_funcs), num_dfs(div_funcs.size()),
            search_params(search_params), results(results),
            jobs_mutex(jobs_mutex), jobs(jobs)
        { }

    virtual ~divcalc_worker() {};

    virtual void do_job(size_t i, size_t j) = 0;

    void operator()();
};

template <typename Distance>
class divcalc_samebags_worker : public divcalc_worker<Distance> {
    typedef divcalc_worker<Distance> super;

    typedef flann::Matrix<typename Distance::ElementType> Matrix;
    typedef flann::Index<Distance> Index;
    typedef std::vector<float> DistVec;
    typedef std::vector<DistVec> DistVecVec;
    typedef std::pair<size_t, size_t> size_pair;

    protected:

    // some crazy C++ template stuff means inherited name won't be auto-resolved
    // see: http://stackoverflow.com/q/4010281/344821
    using super::k;
    using super::dim;
    using super::div_funcs;
    using super::num_dfs;
    using super::search_params;
    using super::results;
    using super::jobs_mutex;
    using super::jobs;

    const Matrix *bags;
    Index **indices;
    const std::vector<DistVec> &rhos;

    public:

    divcalc_samebags_worker(
            const Matrix *bags,
            Index **indices,
            const DistVecVec &rhos,
            const boost::ptr_vector<DivFunc> &div_funcs,
            int k, int dim,
            const flann::SearchParams &search_params,
            flann::Matrix<double> *results,
            boost::mutex &jobs_mutex, std::queue<size_pair> &jobs)
        :
            super(k, dim, div_funcs, search_params, results, jobs_mutex, jobs),
            bags(bags), indices(indices), rhos(rhos)
        { }

    virtual void do_job(size_t i, size_t j);
};

template <typename Distance>
class divcalc_diffbags_worker : public divcalc_worker<Distance> {
    typedef divcalc_worker<Distance> super;

    typedef flann::Matrix<typename Distance::ElementType> Matrix;
    typedef flann::Index<Distance> Index;
    typedef std::vector<float> DistVec;
    typedef std::vector<DistVec> DistVecVec;
    typedef std::pair<size_t, size_t> size_pair;

    protected:

    using super::k;
    using super::dim;
    using super::div_funcs;
    using super::num_dfs;
    using super::search_params;
    using super::results;
    using super::jobs_mutex;
    using super::jobs;

    const Matrix *x_bags, *y_bags;
    Index **x_indices, **y_indices;
    const std::vector<DistVec> &x_rhos, &y_rhos;

    public:
    divcalc_diffbags_worker(
            const Matrix *x_bags, const Matrix *y_bags,
            Index **x_indices, Index **y_indices,
            const DistVecVec &x_rhos, const DistVecVec &y_rhos,
            const boost::ptr_vector<DivFunc> &div_funcs,
            int k, int dim,
            const flann::SearchParams &search_params,
            flann::Matrix<double> *results,
            boost::mutex &jobs_mutex, std::queue<size_pair> &jobs)
        :
            super(k, dim, div_funcs, search_params, results, jobs_mutex, jobs),
            x_bags(x_bags), y_bags(y_bags),
            x_indices(x_indices), y_indices(y_indices),
            x_rhos(x_rhos), y_rhos(y_rhos)
        { }

    virtual void do_job(size_t i, size_t j);
};

////////////////////////////////////////////////////////////////////////////////
// Implementations of the np_divs overloads

template <typename Scalar>
void np_divs(
        const flann::Matrix<Scalar> *bags,
        size_t num_bags,
        flann::Matrix<double>* results,
        const DivParams &params,
        bool ver_alloc)
{
    DEFAULT_DIV_FUNCS

    return np_divs(bags, num_bags, div_funcs, results, params, ver_alloc);
}

template <typename Scalar>
void np_divs(
        const flann::Matrix<Scalar> *bags,
        size_t num_bags,
        const DivFunc &div_func,
        flann::Matrix<double>* results,
        const DivParams &params,
        bool ver_alloc)
{
    boost::ptr_vector<DivFunc> div_funcs;
    div_funcs.push_back(new_clone(div_func));
    return np_divs(bags, num_bags, div_funcs, results, params, ver_alloc);
}


template <typename Scalar>
void np_divs(
        const flann::Matrix<Scalar> *bags,
        size_t num_bags,
        const boost::ptr_vector<DivFunc> &div_funcs,
        flann::Matrix<double>* results,
        const DivParams &params,
        bool ver_alloc)
{
    using std::vector;

    typedef flann::L2<Scalar> Distance;

    typedef flann::Matrix<Scalar> Matrix;
    typedef flann::Index<Distance> Index;
    typedef vector<float> DistVec;

    size_t num_dfs = div_funcs.size();
    size_t dim = bags[0].cols;

    // some setup
    if (ver_alloc)
        verify_allocated(results, num_dfs, num_bags, num_bags);

    int k = params.k;
    if (k < 1)
        throw std::domain_error("np_divs: k < 1 is nonsensical");
    size_t num_threads = get_num_threads(params.num_threads);

    // build kd-trees or whatever
    Index** indices = make_indices<Distance>(
            bags, num_bags, params.index_params);

    // do nearest-neighbor searches for each bag to itself
    const vector<DistVec> &rhos =
        get_rhos(bags, indices, num_bags, k, params.search_params, num_threads);

    // this queue will tell threads what to do
    std::queue<std::pair<size_t, size_t> > jobs;

    // to avoid simultaneous access to jobs. the only other non-const things
    // are the indices (which are thread-safe for searching) and results, which
    // is fine since the threads only touch separate parts of it.
    boost::mutex jobs_mutex;

    // compute away!
    if (num_threads == 1) {
        divcalc_samebags_worker<Distance> worker(
                bags, indices, rhos, div_funcs, k, dim, params.search_params,
                results, jobs_mutex, jobs
        );

        // ignore the queue, just use do_job directly
        for (size_t i = 0; i < num_bags; i++)
            for (size_t j = 0; j <= i; j++)
                worker.do_job(i, j);

    } else {
        // put jobs in the queue
        for (size_t i = 0; i < num_bags; i++)
            for (size_t j = 0; j <= i; j++)
                jobs.push(std::pair<size_t, size_t>(i, j));

        // we keep the worker objects in this ptr_vector so
        // that they don't get copied but also have the correct lifetime
        boost::ptr_vector<divcalc_samebags_worker<Distance> > workers;
        boost::thread_group worker_threads;

        for (size_t i = 0; i < num_threads; i++) {
            // create the worker
            workers.push_back(new divcalc_samebags_worker<Distance>(
                bags, indices, rhos, div_funcs, k, dim, params.search_params,
                results, jobs_mutex, jobs
            ));
            worker_threads.create_thread(boost::ref(workers[i]));
        }
        worker_threads.join_all();
    }

    free_indices(indices, num_bags);
}


template <typename Scalar>
void np_divs(
        const flann::Matrix<Scalar> *x_bags, size_t num_x,
        const flann::Matrix<Scalar> *y_bags, size_t num_y,
        const DivFunc &div_func,
        flann::Matrix<double>* results,
        const DivParams &params,
        bool ver_alloc)
{
    boost::ptr_vector<DivFunc> div_funcs;
    div_funcs.push_back(new_clone(div_func));
    return np_divs(x_bags, num_x, y_bags, num_y, div_funcs, results, params,
                   ver_alloc);
}


template <typename Scalar>
void np_divs(
        const flann::Matrix<Scalar> *x_bags, size_t num_x,
        const flann::Matrix<Scalar> *y_bags, size_t num_y,
        flann::Matrix<double>* results,
        const DivParams &params,
        bool ver_alloc)
{
    DEFAULT_DIV_FUNCS

    return np_divs(x_bags, num_x, y_bags, num_y, div_funcs, results, params,
                   ver_alloc);
}


template <typename Scalar>
void np_divs(
    const flann::Matrix<Scalar>* x_bags, size_t num_x,
    const flann::Matrix<Scalar>* y_bags, size_t num_y,
    const boost::ptr_vector<DivFunc> &div_funcs,
    flann::Matrix<double>* results,
    const DivParams &ps,
    bool ver_alloc)
{   /* Calculates the matrix of divergences between x_bags and y_bags for
     * each of the passed div_funcs, and writes them into the preallocated
     * array of matrices (div_funcs.size() bags, each with num_x rows and
     * num_y cols) to the passed div_funcs. Rows of each matrix are an x_bag,
     * columns are a y_bag.
     *
     * Runs on num_threads threads; if num_threads is 0 (the default), uses one
     * thread per core/hyperthreading unit, as determined by
     * boost::thread::hardware_concurrency (or 1 if that information is
     * unavailable). Note that if you use boost 1.34 or lower, this function is
     * unavailable and so num_threads will always default to 1. If num_threads
     * is 1, doesn't actually spawn any new threads.
     *
     * By default, conducts a quick check that the result matrices were
     * allocated properly; if you're sure that you did and want to skip this
     * check, pass verify_results_alloced=false.
     */

    using std::vector;

    typedef flann::L2<Scalar> Distance;

    typedef flann::Matrix<Scalar> Matrix;
    typedef flann::Index<Distance> Index;
    typedef vector<float> DistVec;

    // save work if we're actually comparing bags to themselves
    if (y_bags == NULL || y_bags == x_bags)
        return np_divs(x_bags, num_x, div_funcs, results, ps, ver_alloc);

    // initial setup work
    size_t num_dfs = div_funcs.size();
    size_t dim = x_bags[0].cols;
    // TODO: check that y_bags[0] (all bags?) is the same dimensions

    int k = ps.k;
    if (k < 1)
        throw std::domain_error("np_divs: k < 1 is nonsensical");
    size_t num_threads = get_num_threads(ps.num_threads);

    if (ver_alloc)
        verify_allocated(results, num_dfs, num_x, num_y);

    // build kd trees or whatever
    Index** x_indices = make_indices<Distance>(x_bags, num_x, ps.index_params);
    Index** y_indices = make_indices<Distance>(y_bags, num_y, ps.index_params);

    // do nearest-neighbor searches for each bag to itself
    const vector<DistVec> &x_rhos =
           get_rhos(x_bags, x_indices, num_x, k, ps.search_params, num_threads);
    const vector<DistVec> &y_rhos =
           get_rhos(y_bags, y_indices, num_y, k, ps.search_params, num_threads);

    // compute the divergences!
    //
    // TODO - check that we actually need nu_y

    // this queue will tell threads what to do
    std::queue<std::pair<size_t, size_t> > jobs;

    // to avoid simultaneous access to jobs. the only other non-const things
    // are the indices (which are thread-safe for searching) and results, which
    // is fine since the threads only touch separate parts of it.
    boost::mutex jobs_mutex;

    if (num_threads <= 1) {
        divcalc_diffbags_worker<Distance> worker(
            x_bags, y_bags, x_indices, y_indices, x_rhos, y_rhos,
            div_funcs, k, dim, ps.search_params, results, jobs_mutex, jobs
        );

        // forget the queue and lock, just do_job directly
        for (size_t i = 0; i < num_x; i++)
            for (size_t j = 0; j < num_y; j++)
                worker.do_job(i, j);

    } else {
        // queue up our jobs
        for (size_t i = 0; i < num_x; i++)
            for (size_t j = 0; j < num_y; j++)
                jobs.push(std::pair<size_t, size_t>(i, j));

        // launch worker threads
        // we keep the worker objects in this ptr_vector so
        // that they don't get copied but also have the correct lifetime
        boost::ptr_vector<divcalc_diffbags_worker<Distance> > workers;
        boost::thread_group worker_threads;

        for (size_t i = 0; i < num_threads; i++) {
            // create the worker
            workers.push_back(new divcalc_diffbags_worker<Distance>(
                x_bags, y_bags, x_indices, y_indices, x_rhos, y_rhos,
                div_funcs, k, dim, ps.search_params, results, jobs_mutex, jobs
            ));
            worker_threads.create_thread(boost::ref(workers[i]));
        }

        worker_threads.join_all();
    }

    free_indices(x_indices, num_x);
    free_indices(y_indices, num_y);
}

////////////////////////////////////////////////////////////////////////////////
// Worker class implementations

template <typename Distance>
void divcalc_worker<Distance>::operator()() {
    size_pair job;
    while (true) {
        { // lock applies only in this scope
            boost::mutex::scoped_lock the_lock(jobs_mutex);

            if (jobs.size() == 0)
                return;

            job = jobs.front();
            jobs.pop();
        }

        this->do_job(job.first, job.second);
    }
}

template <typename Distance>
void divcalc_samebags_worker<Distance>::do_job(size_t i, size_t j) {
    if (i == j) {
        const Matrix &bag = bags[i];
        Index &index = *indices[i];
        const DistVec &rho = rhos[i];

        const DistVec &nu = DKN<Distance, float>(index, bag, k, search_params);

        for (size_t df = 0; df < num_dfs; df++) {
            results[df][i][i] = div_funcs[df](rho, nu, rho, nu, dim, k);
        }
    } else {
        const Matrix  &x_bag = bags[i],       &y_bag = bags[j];
        Index         &x_index = *indices[i], &y_index = *indices[j]; 
        const DistVec &rho_x = rhos[i],       &rho_y = rhos[j];

        const DistVec &nu_x = DKN<Distance, float>(y_index, x_bag, k, search_params);
        const DistVec &nu_y = DKN<Distance, float>(x_index, y_bag, k, search_params);

        for (size_t df = 0; df < num_dfs; df++) {
            const DivFunc &div_func = div_funcs[df];
            results[df][i][j] = div_func(rho_x, nu_x, rho_y, nu_y, dim, k);
            results[df][j][i] = div_func(rho_y, nu_y, rho_x, nu_x, dim, k);
        }
    }
}

template <typename Distance>
void divcalc_diffbags_worker<Distance>::do_job(size_t i, size_t j) {
    const Matrix  &x_bag = x_bags[i],        &y_bag = y_bags[j];
    Index         &x_index = *x_indices[i],  &y_index = *y_indices[j];
    const DistVec &rho_x = x_rhos[i],        &rho_y = y_rhos[j];

    // compute away
    const DistVec &nu_x = DKN<Distance, float>(y_index, x_bag, k, search_params);
    const DistVec &nu_y = DKN<Distance, float>(x_index, y_bag, k, search_params);

    for (size_t df = 0; df < num_dfs; df++) {
        results[df][i][j] = div_funcs[df](rho_x, nu_x, rho_y, nu_y, dim, k);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Helper implementations


size_t get_num_threads(size_t num_threads) {
#if BOOST_VERSION >= 103500
    if (num_threads == 0)
        num_threads = boost::thread::hardware_concurrency();
#endif
    return num_threads > 0 ? num_threads : 1;
}

template <typename T>
void verify_allocated(
        flann::Matrix<T> *matrices, size_t num_matrices,
        size_t rows, size_t cols)
{
    for (size_t i = 0; i < num_matrices; i++) {
        const flann::Matrix<T> &m = matrices[i];
        if (m.rows != rows || m.cols != cols) {
            boost::format err =
                boost::format("expected matrix %d to be %dx%d; it's %d%d")
                % i % rows % cols % m.rows % m.cols;
            std::cerr << err << std::endl;
            throw std::length_error(err.str());
        }
    }
}

template <typename Distance>
flann::Index<Distance>** make_indices(
        const flann::Matrix<typename Distance::ElementType> *datasets,
        size_t number,
        const flann::IndexParams index_params)
{
    typedef flann::Index<Distance> Index;

    // malloc to avoid calling constructors
    Index** indices = (Index**) malloc(sizeof(Index*) * number);

    for (size_t i = 0; i < number; i++) {
        Index* idx = new Index(datasets[i], index_params);
        idx->buildIndex();
        indices[i] = idx;
    }

    return indices;
}

template <typename Distance>
void free_indices(flann::Index<Distance>** indices, size_t n) {
    for (size_t i = 0; i < n; i++)
        delete indices[i];
    free(indices);
}


template <typename Distance>
class rho_getter : boost::noncopyable {
    typedef flann::Index<Distance> Index;
    typedef typename Distance::ResultType Scalar;
    typedef flann::Matrix<Scalar> Matrix;
    typedef std::vector<float> DistVec;

    const Matrix * bags;
    Index ** indices;
    int k;
    const flann::SearchParams &search_params;

    std::vector<DistVec> &rhos;
    boost::mutex &rhos_mutex;

    std::queue<size_t> &jobs;
    boost::mutex &jobs_mutex;

    public:
    rho_getter(const Matrix *bags, Index **indices, int k,
            const flann::SearchParams &search_params,
            std::vector<DistVec> &rhos, boost::mutex &rhos_mutex,
            std::queue<size_t> &jobs, boost::mutex &jobs_mutex)
        :
            bags(bags), indices(indices), k(k), search_params(search_params),
            rhos(rhos), rhos_mutex(rhos_mutex),
            jobs(jobs), jobs_mutex(jobs_mutex)
        { }

    void operator()(){
        size_t i;
        while (true) {
            // get a job
            {
                boost::mutex::scoped_lock the_lock(jobs_mutex);

                if (jobs.size() == 0) return;

                i = jobs.front();
                jobs.pop();
            }

            // compute
            const DistVec &rho = DKN<Distance, float>(
                    *indices[i], bags[i], k+1, search_params);

            // write out results
            {
                boost::mutex::scoped_lock the_lock(rhos_mutex);
                rhos[i] = rho;
            }
        }
    };
};


template <typename Distance>
std::vector<std::vector<float> > get_rhos(
        const flann::Matrix<typename Distance::ElementType> *bags,
        flann::Index<Distance> **indices,
        size_t n,
        int k,
        const flann::SearchParams &search_params,
        size_t num_threads)
{
    // TODO - if dimension is small enough, don't thread

    std::vector<std::vector<float> > rhos;

    if (num_threads == 1) {
        rhos.reserve(n);
        for (size_t i = 0; i < n; i++)
            rhos.push_back(DKN<Distance, float>(*indices[i], bags[i], k+1, search_params));

    } else {
        rhos.resize(n);

        boost::ptr_vector<rho_getter<Distance> > workers;
        boost::thread_group worker_threads;
        boost::mutex rhos_mutex, jobs_mutex;

        std::queue<size_t> jobs;
        for (size_t i = 0; i < n; i++)
            jobs.push(i);

        for (size_t i = 0; i < num_threads; i++) {
            workers.push_back(new rho_getter<Distance>(
                        bags, indices, k, search_params,
                        rhos, rhos_mutex, jobs, jobs_mutex
            ));
            worker_threads.create_thread(boost::ref(workers[i]));
        }

        worker_threads.join_all();
    }

    return rhos;
}

} // close namespace
#endif
