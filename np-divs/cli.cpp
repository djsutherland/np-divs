#include "np-divs/np_divs.hpp"
#include "np-divs/matrix_reader.hpp"

#include <boost/program_options.hpp>

#include <flann/util/matrix.h>


namespace po = boost::program_options;


typedef struct s_popts {

} ProgOpts;


bool parse_args(int argc, const char ** argv, ProgOpts& popts);


int main(int argc, const char ** argv) {
    ProgOpts popts;
    if (!parse_args(argc, argv, popts))
        return 1;

    // 

    return 0;
}

bool parse_args(int argc, const char ** argv, ProgOpts& popts) {
    return true;
}
