classdef PQEncoder < handle & featpipem.encoding.GenericEncoder
    %PQENCODER Bag-of-word histogram computation using the PQ method (hard assignment)
    
    properties
        max_comps % -1 for exact
    end
    
    properties(SetAccess=protected)
        codebook_
        kdtree_
        subquant_cluster_count_
        subquant_class_
    end
    
    methods
        function obj = PQEncoder(codebook)
            % set default parameter values
            obj.max_comps = -1;
            
            % setup encoder
            obj.codebook_ = codebook;
            obj.kdtree_ = cell(length(obj.codebook_),1);
            for i = 1:length(obj.codebook_)
                obj.kdtree_{i} = vl_kdtreebuild(obj.codebook_{i});
            end
            
            % calculate code size
            obj.subquant_cluster_count_ = size(obj.codebook_{1},2);
            for i = 2:length(obj.codebook_)
                if size(obj.codebook_{i},2) ~= obj.subquant_cluster_count_
                    error('Some subquantizers contain a different number of centroids than others');
                end
            end
            bytes_per_subquant = ceil(log(obj.subquant_cluster_count_)/log(2));
            
            if bytes_per_subquant <= 8
                obj.subquant_class_ = 'uint8';
            elseif bytes_per_subquant <= 16
                obj.subquant_class_ = 'uint16';
            elseif bytes_per_subquant <= 32
                obj.subquant_class_ = 'uint32';
            else
                error('Could not fit subquantizer code into uint');
            end
        end
        function dim = get_input_dim(obj)
            subquant_dim = size(obj.codebook_{1},1);
            dim = subquant_dim;
            for i = 2:length(obj.codebook_)
                if size(obj.codebook_{i},1) ~= subquant_dim
                    error('Some subquantizers feature dimension are different');
                end
                dim = dim + subquant_dim;
            end
        end
        function dim = get_output_dim(obj)
            dim = length(obj.codebook_);
        end
        function subquant_class = get_output_class(obj)
            subquant_class = obj.subquant_class_;
        end
        code = encode(obj, feats)
    end
    
end

