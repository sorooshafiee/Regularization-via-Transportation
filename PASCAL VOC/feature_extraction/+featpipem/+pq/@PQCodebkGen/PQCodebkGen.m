classdef PQCodebkGen < handle
    %PQCODEBKGEN Generate codebook of visual words using kmeans
    
    properties
        codebook
        subquant_count
        subquant_bits
        maxcomps % maximum number of comparisons when using ANN (-1 = exact)
    end
    
    methods
        function obj = PQCodebkGen(codebook, subquant_count, subquant_bits)
            obj.codebook = codebook;
            obj.subquant_count = subquant_count;
            obj.subquant_bits = subquant_bits;
            obj.maxcomps = -10;
        end
        codebook = train(obj, training_feats)
    end
    
end

