function [feats, frames] = compute(obj, im)
%COMPUTE Summary of this function goes here
%   Detailed explanation goes here

    [frames, feats] = vl_phow(im, 'Verbose', obj.verbose, ...
        'Sizes', obj.sizes, 'Fast', obj.fast, 'step', obj.step, ...
        'Color', obj.color, 'ContrastThreshold', obj.contrast_threshold, ...
        'WindowSize', obj.window_size, 'Magnif', obj.magnif, ...
        'FloatDescriptors', obj.float_descriptors);
    feats = single(feats);
    
    if obj.remove_zero
        % remove zero features
        nz_feat = any(feats, 1);
        
        feats = feats(:, nz_feat);
        frames = frames(:, nz_feat);
    end
    
    if ~isempty(obj.low_proj)
        % dimensionality reduction
        feats = obj.low_proj * feats;
    end
    
    % augmentation
    if obj.aug_frames

        w = size(im, 2);
        h = size(im, 1);        

        % augment with (x,y)
        feats = [feats; frames(1, :) / w - 0.5; frames(2, :) / h - 0.5];
    end
    
end

