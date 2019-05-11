function [ranked_ids, word_count] = query(obj, code)
%QUERY Summary of this function goes here
%   Detailed explanation goes here

    if obj.indexed_code_count_ == 1 || obh.

    subquant_codewords = size(obj.codebook{1},2);
    subquant_count = length(obj.codebook);
    subquant_len = size(obj.codebok{1},1);
    
    if subquant_count ~= length(code)
        error('code should be of same length as number of subquantizers');
    end

%     % compute squared distances between input code and subquantizer
%     % codewords in codebook
%     
%     sqdists = zeros(subquant_count, subquant_codewords);
%     
%     for i = 1:subquant_count
%         startidx = (i-1)*subquant_len + 1;
%         endidx = i*subquant_len;
%         for j = 1:subquant_codewords
%             sqdists(j,i) = sum((code(startidx:endidx) - obj.codebook{i}(:,j)).^2);
%         end
%     end
    
    % compute intersection (# subquantizer words in common) between
    % input code and test features
    
    postings = cell(subquant_count,1);
    
    for qi = 1:subquant_count
        postings{qi} = obj.invidx_{qi,code(qi)};
    end
    
    commonvws = uint8(zeros(obj.indexed_code_count_,1));
    
    for qi = 1:subquant_count
        procidx = obj.invidx_{qi,code(qi)};
        commonvws(procidx) = commonvws(procidx) + 1;
    end
    
    [word_count, ranked_ids] = sort(commonvws);
    
end

