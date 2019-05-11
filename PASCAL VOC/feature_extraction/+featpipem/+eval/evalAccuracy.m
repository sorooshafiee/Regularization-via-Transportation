function [acc, confus] = evalAccuracy(imdb, scoremat, testset)
%EVALPRECREC Evaluate a dataset using average accuracy
%   Note that this evaluation method requires that only a single class is
%   assigned per image in the ground truth (to allow the confusion matrix
%   to be computed). For example, this is true of the caltech-101 dataset,
%   but not of the VOC datasets. The first output parameter gives the
%   accuracy acheived per class, and the second output, if used, gives the
%   full confusion matrix.

% prepare estimated labels and ground truth labels
% -------------------------------------------------

% compute estimated labels from scoremat
fprintf('Computing estimated labels from scoremat...\n');
[estLabel, estLabel] = max(scoremat, [], 1); %#ok<ASGLU>

% get ground truth labels in form of a cell array of length C
[gt set_size] = featpipem.utility.getImdbGT(imdb, {testset});

% copy across to a vector of length set_size, with a single label
% associated with each image
fprintf('Extracting single-class ground truth labels from IMDB...\n');
gtLabel = ones(1,set_size)*-1;
for i = 1:set_size
    for ci = 1:length(gt)
        % check if current class exists in current image
        if ismember(i,gt{ci})
            if gtLabel(i) == -1
                gtLabel(i) = ci;
            else
                error(['Multiple classes in ground truth for image %d ' ...
                    '- this is not allowed when evaluating using accuracy'], i);
            end
        end
    end
    if gtLabel(i) == -1
        error('No class instance found in ground truth for image %d', i);
    end
end

nClasses = length(imdb.classes.name);

% compute confusion matrix
% -------------------------------------------------

fprintf('Computing confusion matrix...\n');
idx = sub2ind([nClasses, nClasses], ...
              gtLabel, estLabel);
confus = zeros(nClasses);
confus = vl_binsum(confus, ones(size(idx)), idx);

% normalize confusion matrix
fprintf('Normalizing confusion matrix and computing accuracies...\n');
for i = 1:nClasses
    image_count = sum(gtLabel == i);
    for j = 1:nClasses
        if image_count > 0
            confus(i,j) = confus(i,j)/image_count;
        end
    end
end

% calculate accuracies
acc = zeros(nClasses,1);

for ci = 1:nClasses
    acc(ci) = confus(ci,ci);
end

fprintf('DONE\n');

end

