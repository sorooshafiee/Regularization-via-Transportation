function load_proj(obj, fname)
%LOAD_PROJ Summary of this function goes here
%   Detailed explanation goes here

load(fname, 'proj');
if ~isempty(proj)
    obj.proj = proj;
else
    error('No projection found in file!');
end

end

