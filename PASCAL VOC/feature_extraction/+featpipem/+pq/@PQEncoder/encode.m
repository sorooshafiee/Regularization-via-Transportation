function code = encode(obj, feats)
%ENCODE Encode features using the VQ method (hard assignment)

    % Initialize output matrix --------------------------------------------

    code = cast(zeros(length(obj.codebook_), size(feats,2)), obj.subquant_class_);

    % Apply encoding ------------------------------------------------------
    
    % iterate through subquantizers
    startidx = 1;
    endidx = size(obj.codebook_{1},1);
    for qi = 1:length(obj.codebook_)
        if obj.max_comps ~= -1
            % using ann...
            code(qi,:) = vl_kdtreequery(obj.kdtree_{qi}, obj.codebook_{qi}, feats(startidx:endidx,:), ...
                'MaxComparisons', obj.max_comps);
        else
            % using exact assignment...
            [code(qi,:), code(qi,:)] = min(vl_alldist(obj.codebook_{qi}, feats(startidx:endidx,:)), [], 1); %#ok<ASGLU>
        end
        
        startidx = startidx + size(obj.codebook_{qi},1);
        if qi ~= length(obj.codebook_)
            endidx = endidx + size(obj.codebook_{qi+1},1);
        end
    end

end

