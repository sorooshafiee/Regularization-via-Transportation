%  Copyright (c) 2014, Karen Simonyan
%  All rights reserved.
%  This code is made available under the terms of the BSD license (see COPYING file).

classdef IterDSiftExtractor < handle
    %IterDSiftExtractor Feature extractor using iterative dsift
    
    properties
        scale
        scale_factor        
        num_scales
        step
        patch_size
        
        % dimensionality reducing projection
        low_proj
        
        % augment features with their spatial coordinates
        aug_frames
        
        % rootSIFT
        sqrt_map
    end
    
    methods
        function obj = IterDSiftExtractor(varargin)
            
            obj.scale = 0.5;

            obj.scale_factor = 2 ^ (1/2);

            obj.num_scales = 7;
            
            obj.step = 3;
            
            obj.patch_size = 32;
    
            obj.low_proj = [];            

            obj.aug_frames = true;

            obj.sqrt_map = true;
            
            featpipem.utility.set_class_properties(obj, varargin);

        end
        
        [feats, frames] = compute(obj, im)
    end
    
end

