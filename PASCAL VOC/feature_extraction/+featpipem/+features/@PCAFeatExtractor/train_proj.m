function train_proj(obj, imlist)
%TRAIN_PROJ Summary of this function goes here
%   Detailed explanation goes here

% -------------------------------------------------------------------------
% 1. Extract features for training into 'feats' matrix
%     applying any limits on number of features/images
% -------------------------------------------------------------------------

% if trainimage_limit was not left at it's default value
% (indicating all detected images should be used for training)
% select a subset of the images
if obj.trainimage_limit > 0
    idxs = 1:length(imlist);
    idxs = vl_colsubset(idxs, obj.trainimage_limit);
    imlist = imlist(idxs);
end

if obj.descount_limit > 0
    % set truncation value for image features just a little bit
    % larger than descount_limit, so if there are any images
    % with fewer than descount_limit/numImages we still have
    % some chance of getting descount_limit descriptors in the end
    img_descount_limit = ceil(obj.descount_limit / ...
        length(imlist) * 1.1);
    fprintf('Extracting a maximum of %d features from each image...\n', ...
        img_descount_limit);
end

feats = cell(length(imlist),1);

% iterate through images, computing features
pfImcount = length(imlist);
parfor ii = 1:length(imlist)
    fprintf('Computing features for: %s %f %% complete\n', ...
        imlist{ii}, ii/pfImcount*100.00);

    im = imread(imlist{ii});
    im = featpipem.utility.standardizeImage(im);
    feats_all = obj.featextr.compute(im); %#ok<PFBNS>
    
    % if a descount limit applies, discard a fraction of features now to
    % save memory
    if obj.descount_limit > 0
        feats{ii} = vl_colsubset(feats_all, img_descount_limit);
    else
        feats{ii} = feats_all;
    end
end
clear feats_all;
% concatenate features into a single matrix
feats = cat(2, feats{:});

disp('Doing PCA...');

featNorms = sum(feats.^2);
X = vl_colsubset(feats(:,featNorms > 0.1),10000);
X = bsxfun(@minus, X, mean(X,2));
[U,S,V] = svd(X); %#ok<NASGU,ASGLU>

obj.proj = U(:,1:obj.out_dim)'; %do PCA

end

