function pcode = compute(obj, imsize, feats, frames)
%COMPUTE Pool features using the spatial pyramid match kernel
  
    pcode =  feats;
    
    % now post-normalize whole code
    if strcmp(obj.post_norm_type,'l2')
        pcode = pcode/norm(pcode,2);
    elseif strcmp(obj.post_norm_type,'l1')
        pcode = pcode/norm(pcode,1);
    end
end

