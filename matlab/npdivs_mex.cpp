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


/* A MATLAB interface to the C++ NPDivs function.
 */

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <boost/bind.hpp>
#include <boost/exception/all.hpp>
#include <boost/format.hpp>
#include <boost/ptr_container/ptr_vector.hpp>

#include <mex.h>

#include <svm.h>

#include <flann/flann.hpp>

#include <np-divs/matrix_arrays.hpp>
#include <np-divs/div-funcs/from_str.hpp>

#include "sdm/sdm.hpp"
#include "sdm/kernels/gaussian.hpp"
#include "sdm/kernels/linear.hpp"
#include "sdm/kernels/polynomial.hpp"

typedef flann::Matrix<float> MatrixF;
typedef flann::Matrix<double> MatrixD;

using std::string;
using std::vector;

using npdivs::DivParams;

using sdm::SDM;
typedef SDM<float> SDMF;

////////////////////////////////////////////////////////////////////////////////
// Helpers to convert from MATLAB to C++ types

string get_string(const mxArray *thing, const char* err_msg) {
    if (mxIsChar(thing) != 1 || mxGetM(thing) != 1)
        mexErrMsgTxt(err_msg);

    char* c_str = mxArrayToString(thing);
    string str(c_str);
    mxFree(c_str);
    return str;
}

double get_double(const mxArray *thing, const char* err_msg) {
    if (mxIsNumeric(thing) != 1 || mxGetNumberOfElements(thing) != 1)
        mexErrMsgTxt(err_msg);
    return mxGetScalar(thing);
}

size_t get_size_t(const mxArray *thing, const char *err_msg) {
    double d = get_double(thing, err_msg);
    if (d < 0) mexErrMsgTxt(err_msg);
    size_t s = (size_t) (d + .1);
    if (std::abs(d - s) > 1e-10)
        mexErrMsgTxt(err_msg);
    return s;
}

bool get_bool(const mxArray *thing, const char* err_msg) {
    if (mxGetNumberOfElements(thing) == 1) {
        if (mxIsLogical(thing) == 1)
            return mxGetScalar(thing) != 0;

        if (mxIsNumeric(thing) == 1) {
            double d = mxGetScalar(thing);
            if (d == 0)
                return false;
            else if (d == 1)
                return true;
        }
    }
    mexErrMsgTxt(err_msg);
    return false; // to make compilers happy, but this'll never happen
}


////////////////////////////////////////////////////////////////////////////////
// Helper functions to convert from MATLAB matrices to flann::Matrix

// Copy a MATLAB array of type T into a flann::Matrix<K>.
template <typename T, typename K>
void copyIntoFlann(const mxArray *bag, mwSize rows, mwSize cols,
                   flann::Matrix<K> &target)
{
    const T* bag_data = (T*) mxGetData(bag);

    // copy from column-major source to row-major dest, also cast contents
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            target[i][j] = (K) bag_data[j*rows + i];
}

template <typename K>
void copyMatrixIntoFlann(const mxArray *bag, mwSize r, mwSize c,
                         flann::Matrix<K> &tgt) {
    switch (mxGetClassID(bag)) {
        case mxINT8_CLASS:   copyIntoFlann<int8_T,   K>(bag, r, c, tgt); break;
        case mxUINT8_CLASS:  copyIntoFlann<uint8_T,  K>(bag, r, c, tgt); break;
        case mxINT16_CLASS:  copyIntoFlann<int16_T,  K>(bag, r, c, tgt); break;
        case mxUINT16_CLASS: copyIntoFlann<uint16_T, K>(bag, r, c, tgt); break;
        case mxINT32_CLASS:  copyIntoFlann<int32_T,  K>(bag, r, c, tgt); break;
        case mxUINT32_CLASS: copyIntoFlann<uint32_T, K>(bag, r, c, tgt); break;
        case mxINT64_CLASS:  copyIntoFlann<int64_T,  K>(bag, r, c, tgt); break;
        case mxUINT64_CLASS: copyIntoFlann<uint64_T, K>(bag, r, c, tgt); break;
        case mxSINGLE_CLASS: copyIntoFlann<float,    K>(bag, r, c, tgt); break;
        case mxDOUBLE_CLASS: copyIntoFlann<double,   K>(bag, r, c, tgt); break;
        default: mexErrMsgTxt("unsupported bag type");
    }
}

template <typename K>
flann::Matrix<K> get_matrix(const mxArray *mat, K* data) {
    mwSize r = mxGetM(mat);
    mwSize c = mxGetM(mat);
    flann::Matrix<K> fla(data, r, c);

    copyMatrixIntoFlann(mat, r, c, fla);
    return fla;
}

// Copy a MATLAB cell array of distribution samples (with consistent number
// of columns) into a newly-allocated array of flann::Matrix<K>s.
template <typename K>
flann::Matrix<K> *get_matrix_array(const mxArray *bags, mwSize n,
        bool mat_alloc=true)
{
    typedef flann::Matrix<K> Matrix;

    if (!mxIsCell(bags))
        mexErrMsgTxt("get_matrix_array: non-cell argument");

    Matrix *flann_bags;
    if (mat_alloc)
        flann_bags = (Matrix *) mxCalloc(n, sizeof(Matrix));
    else
        flann_bags = new Matrix[n];

    mwSize rows;
    const mwSize cols = mxGetN(mxGetCell(bags, 0));

    const mxArray *bag;

    for (mwSize i = 0; i < n; i++) {
        bag = mxGetCell(bags, i);

        // check dimensions
        if (mxGetNumberOfDimensions(bag) != 2)
            mexErrMsgTxt("bag has too many dimensions");
        rows = mxGetM(bag);
        if (mxGetN(bag) != cols)
            mexErrMsgTxt("inconsistent number of columns in bags");

        // allocate the result matrix
        K* data;
        if (mat_alloc)
            data = (K*) mxCalloc(rows * cols, sizeof(K));
        else
            data = new K[rows * cols];
        flann_bags[i] = Matrix(data, rows, cols);

        copyMatrixIntoFlann<K>(bag, rows, cols, flann_bags[i]);
    }

    return flann_bags;
}

template <typename T>
flann::Matrix<T>* matalloc_matrix_array(size_t n, size_t rows, size_t cols) {
    typedef flann::Matrix<T> Matrix;
    Matrix* array = (Matrix*) mxCalloc(n, sizeof(Matrix));

    for (size_t i = 0; i < n; i++) {
        array[i] = Matrix((T*) mxCalloc(rows*cols, sizeof(T)), rows, cols);
    }
    return array;
}

void free_matalloced_matrix_array(flann::Matrix_ *bags, mwSize n) {
    for (mwSize i = 0; i < n; i++)
        mxFree(bags[i].ptr());
    mxFree(bags);
}


////////////////////////////////////////////////////////////////////////////////
// Helpers to convert from C++ to MATLAB types

template <typename T> struct matlab_classid {
    mxClassID mxID;
    matlab_classid() : mxID(mxUNKNOWN_CLASS) {}
};
template<> matlab_classid<int8_T>  ::matlab_classid() : mxID(mxINT8_CLASS) {}
template<> matlab_classid<uint8_T> ::matlab_classid() : mxID(mxUINT8_CLASS) {}
template<> matlab_classid<int16_T> ::matlab_classid() : mxID(mxINT16_CLASS) {}
template<> matlab_classid<uint16_T>::matlab_classid() : mxID(mxUINT16_CLASS) {}
template<> matlab_classid<int32_T> ::matlab_classid() : mxID(mxINT32_CLASS) {}
template<> matlab_classid<uint32_T>::matlab_classid() : mxID(mxUINT32_CLASS) {}
template<> matlab_classid<int64_T> ::matlab_classid() : mxID(mxINT64_CLASS) {}
template<> matlab_classid<uint64_T>::matlab_classid() : mxID(mxUINT64_CLASS) {}
template<> matlab_classid<float>   ::matlab_classid() : mxID(mxSINGLE_CLASS) {}
template<> matlab_classid<double>  ::matlab_classid() : mxID(mxDOUBLE_CLASS) {}

// make a MATLAB matrix from a vector<vector<T>>
// assumes the inner vectors are of equal length
template <typename T>
mxArray *make_matrix(const vector< vector<T> > vec_matrix) {
    mwSize m = vec_matrix.size();
    mwSize n = m > 0 ? vec_matrix[0].size() : 0;

    mxClassID id = matlab_classid<T>().mxID;
    mxArray* mat = mxCreateNumericMatrix(m, n, id, mxREAL);
    T* data = (T*) mxGetData(mat);

    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++)
            data[i + j*m] = vec_matrix[i][j];
    return mat;
}

// make a MATLAB matrix from a flann::Matrix<T>
template <typename T>
mxArray *make_matrix(const flann::Matrix<T> bag) {
    mxClassID id = matlab_classid<T>().mxID;
    mxArray* mat = mxCreateNumericMatrix(bag.rows, bag.cols, id, mxREAL);
    T* data = (T*) mxGetData(mat);

    for (size_t i = 0; i < bag.rows; i++)
        for (size_t j = 0; j < bag.cols; j++)
            data[i + j*bag.rows] = bag[i][j];
    return mat;
}

// make a MATLAB cell vector of matrices
template <typename T>
mxArray *make_matrix_cells(const flann::Matrix<T> *bags, size_t n) {
    mxArray *cells = mxCreateCellMatrix(1, n);

    for (size_t i = 0; i < n; i++)
        mxSetCell(cells, i, make_matrix(bags[i]));

    return cells;
}

////////////////////////////////////////////////////////////////////////////////
// Function to print a progress bar

class ProgressBar {
    size_t total;

    public:

    ProgressBar(size_t total) : total(total) { }

    void update(size_t left) {
        size_t done = total - left;
        size_t percent = (size_t) floor(done * 100. / total);

        mexPrintf("\r%d / %d [%d%%]           ", done, total, percent);
        if (left == 0)
            mexPrintf("\n");
        //mexCallMATLAB(0, NULL, 0, NULL, "drawnow");
        //mexEvalString("drawnow");
    }
};


////////////////////////////////////////////////////////////////////////////////
// Function to compute divergences

struct DivOptions {
    vector<string> div_funcs;
    int k;
    size_t num_threads;
    string index_type;
    bool show_progress;

    DivOptions() :
        k(3), num_threads(0), index_type("kdtree")
    {}

    void parseOpt(string name, mxArray* val) {
        if (name == "div_funcs") {
            if (!mxIsCell(val))
                mexErrMsgTxt("div_funcs must be a cell array of strings");
            mwSize nel = mxGetNumberOfElements(val);

            for (mwSize i = 0; i < nel; i++) {
                const mxArray *v = mxGetCell(val, i);
                div_funcs.push_back(get_string(v,
                        "div_funcs must be a cell array of strings"));
            }

        } else if (name == "k") {
            k = get_size_t(val, "k must be a positive integer");
            if (k < 1)
                mexErrMsgTxt("k must be a positive integer");

        } else if (name == "num_threads") {
            num_threads = get_size_t(val,
                    "num_threads must be a nonnegative integer");

        } else if (name == "index") {
            index_type = get_string(val, "index must be a string");

        } else if (name == "show_progress") {
            show_progress = get_bool(val, "show_progress must be a boolean");

        } else {
            mexErrMsgTxt(("unknown divs option: " + name).c_str());
        }
    }

    DivParams getDivParams(const ProgressBar &pbar) const {
        flann::SearchParams search_params(-1);

        return DivParams(k,
                npdivs::index_params_from_str(index_type),
                search_params,
                num_threads,
                show_progress ? 200 : 0,
                boost::bind(&ProgressBar::update, pbar, _1));
    }
};


void do_divs(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
    if (nrhs != 3) mexErrMsgTxt("npdivs takes exactly three arguments");
    if (nlhs != 1) mexErrMsgTxt("npdivs returns exactly 1 output");

    const mxArray *x_bags_m = prhs[0];
    const mxArray *y_bags_m = prhs[1];
    const mxArray *opts_m = prhs[2];

    // x_bags; alloc with matlab, because they shouldn't persist
    mwSize num_x = mxGetNumberOfElements(x_bags_m);
    MatrixF *x_bags = get_matrix_array<float>(x_bags_m, num_x, true);

    // y_bags, if necessary
    mwSize num_y = num_x;
    MatrixF *y_bags = NULL;

    if (y_bags_m != NULL && y_bags_m != x_bags_m) {
        num_y = mxGetNumberOfElements(y_bags_m);
        if (num_y == 0) {
            num_y = num_x;
        } else {
            y_bags = get_matrix_array<float>(y_bags_m, num_y, true);
        }
    }

    // third argument: options
    if (!mxIsStruct(opts_m) || mxIsEmpty(opts_m))
        mexErrMsgTxt("np_divs options must be a struct");
    mwSize nfields = mxGetNumberOfFields(opts_m);
    mwSize nels = mxGetNumberOfElements(opts_m);
    if (nels != 1) {
        mexErrMsgTxt("options should be a single struct, not struct array");
    }

    DivOptions opts;
    for (mwSize i = 0; i < nfields; i++) {
        opts.parseOpt(string(mxGetFieldNameByNumber(opts_m, i)),
                      mxGetFieldByNumber(opts_m, 0, i));
    }

    size_t num_df = opts.div_funcs.size();
    if (num_df == 0) {
        opts.div_funcs.push_back("l2");
        num_df = 1;
    }

    boost::ptr_vector<npdivs::DivFunc> dfs;
    for (size_t i = 0; i < num_df; i++)
        dfs.push_back(npdivs::div_func_from_str(opts.div_funcs[i]));

    // allocate space for results
    MatrixD *divs = matalloc_matrix_array<double>(num_df, num_x, num_y);


    // run it!
    ProgressBar pbar(y_bags == NULL ? (num_x+1) * num_x / 2 : num_x * num_y);

    npdivs::np_divs(x_bags, num_x, y_bags, num_y,
            dfs, divs, opts.getDivParams(pbar));

    // copy into output
    mxArray* divs_cell = make_matrix_cells(divs, num_df);

    // kill temp vars
    free_matalloced_matrix_array(divs, num_df);

    plhs[0] = divs_cell;
}

void mexFunction(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
    try {
        do_divs(nlhs, plhs, nrhs, prhs);
    } catch (boost::exception &e) {
        mexPrintf("\nerror: %s", boost::diagnostic_information(e).c_str());
        throw;
    } catch (std::exception &e) {
        mexPrintf("\nerror: %s", boost::diagnostic_information(e).c_str());
        throw;
    }
}
