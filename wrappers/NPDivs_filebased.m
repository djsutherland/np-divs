function [Ds] = NPDivs_filebased(x_bags, y_bags, div_funcs, options)
%NPDivs Calculate nonparametric divergence estimates via the npdivs binary.
%
% Arguments:
%   x_bags: a cell array of data matrices (each n_i x D)
%   y_bags: a cell array of data matrices (each n_i x D), or [], meaning
%         the same thing as passing x_bags again (but is more efficient)
%   div_funcs: a cell array of string specifications for divergence
%         functions, such as "l2", "renyi:.99", "alpha:.2:1".
%         Possible functions include l2, alpha, bc, hellinger, renyi.
%         Some support a first argument specifying a parameter: renyi:.99 
%         means the Renyi-.99 divergence. All support a last parameter, 
%         which determines the way that large values are normalized: 
%         bc:.95 or renyi:.99:.95 means certain calculated intermediate
%         values above the 95th percentile are cut down; 1 means not to 
%         do this; default is .99.
%   options: a struct array with the following possible members:
%         cmd_name: the name of the program to call.
%              use if npdivs isn't on your PATH.
%         k: the k for k-nearest-neighbor. Default 3.
%         index: the nearest-neighbor index to use. Options include linear,
%              kdtree. Default is kdtree. Use linear for high-dimensional,
%              relatively sparse data.
%         num_threads: the number of threads to use in calculation. 0 (the
%              default) means one per core.
%         resultsfile: the file to save results into.
%              default is a temporary file, which will be deleted.
%
% Returns:
%   Ds: an array of divergence estimates. Ds(i, j, n) is div(x_i, y_j) for
%         the nth divergence function.

if nargin < 2; y_bags = []; end
if nargin < 3; div_funcs = {'l2'}; end
if nargin < 4; options = []; end

do_y = ~isempty(y_bags);

% check that dimensionalities agree
dim = size(x_bags{1}, 2);
for ind = 1:numel(x_bags)
    assert(ndims(x_bags{ind}) == 2);
    assert(size(x_bags{ind}, 1) > 0);
    assert(size(x_bags{ind}, 2) == dim);
end

if do_y
    for ind = 1:numel(y_bags)
        assert(ndims(y_bags{ind}) == 2);
        assert(size(y_bags{ind}, 1) > 0);
        assert(size(y_bags{ind}, 2) == dim);
    end
end

% write our input data out to tempfiles
xfile = tempname();
writebags(x_bags, xfile);

if do_y
    yfile = tempname();
    writebags(y_bags, yfile);
end

% build up our argument string
threads = getopt(options, 'num_threads', 0);
if threads == 0;
    threads = feature('numCores'); % works even if boost is ancient
end

if isfield(options, 'resultsfile')
    del_results = false;
    resultsfile = options.resultsfile;
else
    del_results = true;
    resultsfile = tempname();
end

cmd = [getopt(options, 'cmd_name', 'npdivs') ...
    sprintf(' --num-threads=%d', getopt(options, 'num_threads', 0)) ...
    sprintf(' -k%d', getopt(options, 'k', 3)) ...
    ' -i ' getopt(options, 'i', 'kdtree') ...
    ' -r ' resultsfile ...
    ' -x ' xfile ...
];
if do_y
   cmd = [cmd ' -y ' yfile]; 
end
for ind = 1 : numel(div_funcs)
    cmd = [cmd ' -f ' div_funcs{ind}]; %#ok<AGROW>
end

% call it!
[status, output] = system(cmd); % TODO handle stdout better?
if status
    disp(output)
    delete(xfile, resultsfile); if do_y; delete(yfile); end
    return
end

% parse out our response
Ds = readresult(resultsfile);

% clean up our temp files
delete(xfile);
if del_results; delete(resultsfile); end
if do_y; delete(yfile); end

end

function [val] = getopt(opts, key, default)
    if isfield(opts, key)
        val = opts.(key);
    else
        val = default;
    end
end

function [] = writebags(bags, filename)
    fid = fopen(filename, 'w');
    for ind = 1 : numel(bags)
        bag = bags{ind};
        for i = 1 : size(bag, 1)
            fprintf(fid, '%f', bag(i, 1));
            for j = 2 : size(bag, 2)
                fprintf(fid, ', %f', bag(i, j));
            end
            fprintf(fid, '\n');
        end
        fprintf(fid, '\n');
    end
    fprintf(fid, '\n');
    fclose(fid);
end

function [ary] = readresult(filename)
    fid = fopen(filename);
    ary = {};
    while true
        this_ary = [];
        while true
            line = fgetl(fid);
            if ~ischar(line) || all(isspace(line))
                break
            end
            this_ary(end+1,:) = cell2mat(textscan(line, '%f,'))'; %#ok<AGROW>
        end
        
        ary{end+1} = this_ary; %#ok<AGROW>
        
        if line == -1
            fclose(fid);
            ary = cat(3, ary{:});
            return
        end
    end
end
