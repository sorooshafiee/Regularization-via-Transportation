classdef DeepExtractor < handle & featpipem.features.GenericFeatExtractor
    %PHOWEXTRACTOR Feature extractor for PHOW features
    
    properties
        net
        layer
    end
    
    methods
        function obj = DeepExtractor(varargin)
            obj.net = [];
            obj.layer = [];
            
            featpipem.utility.set_class_properties(obj, varargin);
        end
        [feats, frames] = compute(obj, im)
    end
    
end
