function [feats, frames] = compute(obj, im)
%COMPUTE Summary of this function goes here
%   Detailed explanation goes here

    [feats, frames] = obj.featextr.compute(im);
    
    if ~isempty(obj.proj)
        feats = obj.proj*feats;
    end
    
end

