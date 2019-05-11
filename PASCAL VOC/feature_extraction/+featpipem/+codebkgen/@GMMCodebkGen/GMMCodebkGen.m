classdef GMMCodebkGen < handle & featpipem.codebkgen.GenericCodebkGen
    %GMMCodebkGen Generate codebook of visual words using GMM
    
    properties
        cluster_count % number of visual words in codebook
        descount_limit % limit on # features to use for clustering
        trainimage_limit % limit on # images to use for clustering  
        maxcomps % maximum number of comparisons when using ANN (-1 = exact)
        GMM_init % GMM initialisation method
    end
    
    methods
        function obj = GMMCodebkGen(featextr, cluster_count)
            obj.featextr = featextr;
            obj.cluster_count = cluster_count;
            obj.descount_limit = 1e6;
            obj.trainimage_limit = -1;    
            obj.maxcomps = ceil(cluster_count/4);
            obj.GMM_init = 'rand';
        end
        
        codebook = train(obj, imlist, varargin)
    end
    
end

