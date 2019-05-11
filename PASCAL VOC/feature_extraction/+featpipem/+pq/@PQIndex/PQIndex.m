classdef PQIndex < handle
    %PQINDEX Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        codebook
        invidx_
        indexed_code_count_
%         sqdists_
        
    end
    
    methods
        function obj = PQIndex(codebook)
            obj.codebook = codebook;
            
            obj.invidx_ = [];
            obj.indexed_code_count_ = 0;
%             obj.sqdists_ = -1;
%             
%             subquant_codewords = size(obj.codebook{1},2);
%             subquant_count = length(obj.codebook);
            
%             obj.sqdists_ = cell(subquant_count,1);
%             
%             for qi = 1:length(subquant_count)
%                 obj.sqdists_{qi} = featpipem.lib.LTMatrix(subquant_count);
%                 % compute subquantizer codeword-subquantizer codeword squared distances
%                 for j = 1:length(subquant_codewords)
%                     for i = (j+1):length(subquant_codewords)
%                         obj.sqdists_(i,j) = ...
%                             sum((obj.codebook{qi}(:,i)-obj.codebook{qi}(:,j)).^2);
%                     end
%                 end
%             end
            
        end
        
        idxstart = index(obj, codes)
        [ranked_ids, word_count] = query(obj, code)
        
    end
    
end

