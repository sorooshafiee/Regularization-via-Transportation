function pcode = compute(obj, imsize, feats, frames)
%COMPUTE Pool features using the spatial pyramid match kernel
  
    pcode =  obj.encoder_.encode(feats);    
       
    % now normalize all sub-bins
    if strcmp(obj.subbin_norm_type, 'l2')

        n_feat = length(pcode);
        u = pcode(1:fix(n_feat/2));
        v = pcode(fix(n_feat/2)+1:end);
        nrm = sqrt(u.^2 + v.^2);
        u = u ./ (max(nrm, eps));
        v = v ./ (max(nrm, eps));
        pcode = [u; v];
        
    elseif strcmp(obj.subbin_norm_type, 'l1')
        
        n_feat = length(pcode);
        u = pcode(1:fix(n_feat/2));
        v = pcode(fix(n_feat/2)+1:end);
        nrm = abs(u) + abs(v);
        u = u ./ (max(nrm, eps));
        v = v ./ (max(nrm, eps));
        pcode = [u; v];
        
    end
    
    % now normalize whole code
    if strcmp(obj.norm_type,'l2')
        pcode = pcode/norm(pcode,2);
    elseif strcmp(obj.norm_type,'l1')
        pcode = pcode/norm(pcode,1);
    end
    
    % now apply kernel map if specified
    if ~isequal(obj.kermap, 'none')        
        % (note: when adding extra kernel maps, note that the getDim function
        % must also be modified to reflect the appropriate increase in code
        % dimensionality)
        if strcmp(obj.kermap,'homker')
            % chi-squared approximation
            pcode = vl_homkermap(pcode, 1, 'kchi2');
        elseif strcmp(obj.kermap,'hellinger')
            % "generalised" (signed) Hellinger kernel
            pcode = sign(pcode) .* sqrt(abs(pcode));        
        end

        % now post-normalize whole code
        if strcmp(obj.post_norm_type,'l2')
            pcode = pcode/norm(pcode,2);
        elseif strcmp(obj.post_norm_type,'l1')
            pcode = pcode/norm(pcode,1);
        end
    end
end

