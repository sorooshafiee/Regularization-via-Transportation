function codebook = train(obj, training_feats)
%TRAIN Summary of this function goes here
%   Detailed explanation goes here

cluster_count = size(training_feats,1);

if obj.maxcomps == -10
    obj.maxcomps = ceil(cluster_count/4);
end

subquant_cluster_count = 2^obj.subquant_bits;
subquant_ip_len = cluster_count/obj.subquant_count;

if mod(subquant_ip_len,1)
    error('subquant_count must divide codes into integer chunks');
end

codebook = cell(obj.subquant_count,1);

fprintf('Starting training for %d subquantizers of dimensionality %d into %d clusters\n', ...
    obj.subquant_count, subquant_ip_len, subquant_cluster_count);

for qi = 1:obj.subquant_count
    fprintf('Clustering features for subquantizer %d of %d...\n', qi, obj.subquant_count);
    
    startidx = (qi-1)*subquant_ip_len+1;
    endidx = qi*subquant_ip_len;
    
    if obj.maxcomps < 1
        codebook{qi} = vl_kmeans(training_feats(startidx:endidx,:), subquant_cluster_count, ...
            'verbose', 'algorithm', 'elkan');
    else
        codebook{qi} = featpipem.lib.annkmeans(training_feats(startidx:endidx,:), subquant_cluster_count, ...
            'verbose', true, 'MaxNumComparisons', obj.maxcomps, ...
            'MaxNumIterations', 150);
    end
end

fprintf('Done training subquantizers!\n');

end

