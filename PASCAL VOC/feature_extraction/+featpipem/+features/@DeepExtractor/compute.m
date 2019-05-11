function [feats, frames] = compute(obj, im)
%COMPUTE Summary of this function goes here
%   Detailed explanation goes here
    im = im * 255.0;
    im_ = imresize(im, [256, 256]);
%     if size(im,1) < size(im,2)
%         im_ = imresize(im, [obj.net.meta.normalization.imageSize(1), NaN]);
%     else
%         im_ = imresize(im, [NaN, obj.net.meta.normalization.imageSize(2)]);
%     end
    % crop from center
    [p3, p4, ~] = size(im_);
    q1 = obj.net.meta.normalization.imageSize(1);
    i3_start = floor((p3-q1)/2);
    i3_stop = i3_start + q1;

    i4_start = floor((p4-q1)/2);
    i4_stop = i4_start + q1;
    
    im_ = im_(i3_start+1:i3_stop, i4_start+1:i4_stop, :);
    
    im_ = im_ - obj.net.meta.normalization.averageImage ;
    res = vl_simplenn(obj.net, im_) ;
    feats = squeeze(res(obj.layer).x);
    frames = [];
    
end

