function low_proj = loaddimredextr(pcafeatextr, prms)
%loaddimred Summary of this function goes here
%   Detailed explanation goes here

if exist(prms.dimred,'file')
    pcafeatextr.load_proj(prms.dimred);
else
    % only train PCA reduction if output dimension is less than or equal to
    % original feature dimension
    if pcafeatextr.featextr.out_dim >= pcafeatextr.out_dim
        trainval_files = prms.imdb.images.name(...
                        prms.imdb.images.set == prms.imdb.sets.TRAIN | ...
                        prms.imdb.images.set == prms.imdb.sets.VAL);

        num_files = numel(trainval_files);

        imfiles = cell(num_files, 1);

        for i = 1:num_files
            imfiles{i} = fullfile(prms.paths.dataset, trainval_files{i});
        end

        % do training...
        pcafeatextr.train_proj(imfiles);

        pcafeatextr.save_proj(prms.dimred); 
    end
end

end

