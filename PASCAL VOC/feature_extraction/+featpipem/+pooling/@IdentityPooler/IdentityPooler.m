classdef IdentityPooler < handle & featpipem.pooling.GenericPooler
    %SELDMPooler Pooling using the spatial extended local descriptors method
    
    properties
        n_dim             % feature dimension
        post_norm_type    % 'l1' or 'l2' (or other value = none)
        kermap            % 'homker', 'hellinger' (or other value = none [default])
    end
    
    methods
        function obj = IdentityPooler()
            % set default parameter values
            obj.n_dim = 'none';
            obj.post_norm_type = 'none';
            obj.kermap = 'none';
        end
        function dim = get_output_dim(obj)
            dim = obj.n_dim;
        end     
        pcode = compute(obj, imsize, feats, frames)
    end
    
end

