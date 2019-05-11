function save_proj(obj, fname)
%SAVE_PROJ Summary of this function goes here
%   Detailed explanation goes here

proj = obj.proj;
if ~isempty(proj)
    save(fname, 'proj');
else
    error('No projection computed!');
end

end

