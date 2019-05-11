classdef PCAFeatExtractor < handle & featpipem.features.GenericFeatExtractor
    %PCAFEATEXTRACTOR Class wrapping other feature extractor with PCA dim
    %reduction
    
    properties(SetAccess = protected)
        featextr
        %out_dim
        descount_limit % limit on # features to use for training
        trainimage_limit % limit on # images to use for training
        proj
    end
    
    methods
        function obj = PCAFeatExtractor(featextr, out_dim)
            obj.featextr = featextr;
            obj.out_dim = out_dim;
            obj.descount_limit = 1e6;
            obj.trainimage_limit = -1;
            proj = [];
        end
        train_proj(obj, imlist, varargin)
        save_proj(obj, fname)
        load_proj(obj, fname)
        [feats, frames] = compute(obj, im)
    end
    
end

