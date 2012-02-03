#include "np-divs/matrix_io.hpp"
#include "np-divs/matrix_arrays.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/algorithm/string/trim.hpp>
#include <boost/format.hpp>
#include <boost/tokenizer.hpp>

#include <flann/util/matrix.h>

namespace NPDivs {

using boost::algorithm::trim;
using std::atof;
using std::domain_error;
using std::istream;
using std::string;
using std::vector;
using flann::Matrix;

typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;

vector< vector<double> > matrix_vector_from_csv(istream &in, size_t dim) {
    /* Reads a matrix of doubles from CSV-style input, stopping once a
     * blank line has been consumed.
     *
     * Throws domain_error and stops reading if the rows are not of length
     * dim; if dim is 0, the length of the first line is used.
     */
    vector< vector<double> > matrix;
    string line;

    while (getline(in, line)) {
        trim(line);
        if (line.empty())
            break;

        Tokenizer tok(line);

        vector<double> row;
        if (dim > 0)
            row.reserve(dim);

        for (Tokenizer::iterator i = tok.begin(); i != tok.end(); i++)
            row.push_back(atof(i->c_str()));

        if (dim == 0)
            dim = row.size();
        else if (dim != row.size())
            throw domain_error((boost::format(
                            "nonrectangular matrix: expected %d cols, got %d")
                        % dim % row.size()).str());

        matrix.push_back(row);
    }

    return matrix;
}

vector< vector< vector<double> > >
matrices_vector_from_csv(istream &in, size_t dim) {
    /* Reads a group of matrices of doubles from CSV-style input, with each
     * matrix separated by a single blank line.
     *
     * Throws domain_error and stops reading as soon as a line of length
     * other than dim is encountered; if dim = 0, the length of the first
     * line is used.
     */
    vector< vector< vector<double> > > matrices;
    while (true) {
        vector< vector<double> > m = matrix_vector_from_csv(in, dim);
        if (m.size() == 0)
            break;
        else if (dim == 0)
            dim = m[0].size();

        matrices.push_back(m);
    }
    return matrices;
}

Matrix<double> matrix_from_csv(istream &in) {
    return vector_to_matrix(matrix_vector_from_csv(in));
}

Matrix<double>* matrices_from_csv(istream &in, size_t &n) {
    vector<vector<vector<double> > > vec = matrices_vector_from_csv(in);
    n = vec.size();
    return vector_to_matrix_array(vec);
}

// instantiations of templates
template void matrix_to_csv(std::ostream&, Matrix<double>);
template void matrix_to_csv(std::ostream&, Matrix<float>);

template void matrix_array_to_csv(std::ostream&, Matrix<double>*, size_t n);
template void matrix_array_to_csv(std::ostream&, Matrix<float>*, size_t n);

} // end namespace
