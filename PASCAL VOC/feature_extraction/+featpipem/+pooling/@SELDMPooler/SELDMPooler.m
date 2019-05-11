classdef SELDMPooler < handle & featpipem.pooling.GenericPooler
    %SELDMPooler Pooling using the spatial extended local descriptors method
    
    properties
        subbin_norm_type    % 'l1' or 'l2' (or other value = none)
        norm_type    % 'l1' or 'l2' (or other value = none)
        post_norm_type    % 'l1' or 'l2' (or other value = none)
        kermap  % 'homker', 'hellinger' (or other value = none [default])
    end
    
    properties(SetAccess=protected)
        encoder_     % implementation of featpipem.encoding.GenericEncoder
    end
    
    methods
        function obj = SELDMPooler(encoder)
            % set default parameter values
            obj.subbin_norm_type = 'none';
            obj.norm_type = 'none';
            obj.post_norm_type = 'none';
            obj.kermap = 'none';
            
            % setup encoder
            obj.encoder_ = encoder;
        end
        function dim = get_output_dim(obj)

            dim = obj.encoder_.get_output_dim();

        end     
        pcode = compute(obj, imsize, feats, frames)
    end
    
end

