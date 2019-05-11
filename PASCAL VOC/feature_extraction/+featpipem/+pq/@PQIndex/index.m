function idxstart = index(obj, codes)
%INDEX Summary of this function goes here
%   Detailed explanation goes here

    subquant_codewords = size(obj.codebook{1},2);
    subquant_count = length(obj.codebook);
    
    idxstart = obj.indexed_code_count_ + 1;
    obj.indexed_code_count_ = obj.indexed_code_count_ + size(codes,2);
    
    % prepare inverted index storage
    
    bitlen = log(obj.indexed_code_count_)/log(2);
    
    if isempty(obj.invidx_)
        obj.invidx_ = cell(subquant_count, subquant_codewords);
    end
    
    if bitlen <= 16
        idxclass = 'uint16';
    elseif bitlen <= 32
        idxclass = 'uint32';
    else
        error('too many codes to fit into inverted index');
    end

    
    
    % iterate through subquantizers
    for i = 1:subquant_count
        % iterate through subquantizer codewords
        for j = 1:subquant_codewords
            % iterate through database codes
            for ci = idxstart:obj.indexed_code_count_
                if codes(i,ci-idxstart+1) == j
                    % add code index to posting list
                    obj.invidx_{i,j} = [obj.invidx_{i,j} cast(ci,idxclass)];
                end
            end
        end
    end
end

