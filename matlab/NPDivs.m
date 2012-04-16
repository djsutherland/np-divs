function [Ds] = NPDivs(x_bags, y_bags, div_funcs, options)
% NPDivs Calculate nonparametric divergence estimates, using the MEX interface
%        to the npdivs library.
%
% If the MEX interface is not available, a file-based interface to the
% binary with a compatible signature is also available (NPDivs_filebased).
%
% Arguments:
%   x_bags: a cell array of data matrices (each n_i x D)
%
%   y_bags: a cell array of data matrices (each n_i x D), or [], meaning the
%           same thing as passing x_bags again (but is more efficient)
%
%    div_funcs: a cell array of string specifications for
%           divergence functions, such as 'l2', 'renyi:.99', 'alpha:.2'.
%
%           Default is {'l2'}.
%
%           Possible functions include l2, alpha, bc, hellinger, renyi, linear.
%
%           Some support an argument specifying a divergence parameter:
%           renyi:.99 means the Renyi-.99 divergence.
%
%   options: a struct array with the following possible members:
%         k: the k for k-nearest-neighbor. Default 3.
%
%         index: the nearest-neighbor index to use. Options are linear, kdtree. 
%              Default is kdtree. Use linear for high-dimensional, relatively
%              sparse data.
%
%         num_threads: the number of threads to use in calculation.
%              0 (the default) means one per core.
%
%         show_progress: whether to show progress as computation occurs.
%              Default: only if the size of each return matrix is > 5,000.

if nargin < 2; y_bags = []; end
if nargin < 3; div_funcs = {'l2'}; end
if nargin < 4; options = struct(); end


% check that dimensionalities agree
dim = size(x_bags{1}, 2);
num_x = numel(x_bags);
for ind = 1:num_x
    assert(ndims(x_bags{ind}) == 2);
    assert(size(x_bags{ind}, 1) > 0);
    assert(size(x_bags{ind}, 2) == dim);
end

if isempty(y_bags)
    num_y = num_x;
else
    num_y = numel(y_bags);
    for ind = 1:num_y
        assert(ndims(y_bags{ind}) == 2);
        assert(size(y_bags{ind}, 1) > 0);
        assert(size(y_bags{ind}, 2) == dim);
    end
end

options.div_funcs = div_funcs;
if ~isfield(options, 'show_progress')
    options.show_progress = num_x * num_y > 5000;
end

Ds = npdivs_mex(x_bags, y_bags, options);
end
