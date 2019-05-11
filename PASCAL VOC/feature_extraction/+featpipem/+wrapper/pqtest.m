function res = pqtest(prms, codebook, featextr, encoder, pooler, classifier)
%TEST Summary of this function goes here
%   Detailed explanation goes here

% --------------------------------
% Prepare output filenames
% --------------------------------

trainSetStr = [];
for si = 1:length(prms.splits.train)
    trainSetStr = [trainSetStr prms.splits.train{si}]; %#ok<AGROW>
end

testSetStr = [];
for si = 1:length(prms.splits.test)
    testSetStr = [testSetStr prms.splits.test{si}]; %#ok<AGROW>
end

kChunkIndexFile = fullfile(prms.paths.codes, sprintf('%s_chunkindex.mat', prms.experiment.codes_suffix));
kPQCodebookFile = prms.pqcodebook;
kPQCodesFile = fullfile(prms.paths.codes, sprintf('%s_pq%scodes.mat', prms.experiment.codes_suffix, prms.experiment.pq_tag));
kPQInvIdxFile = fullfile(prms.paths.compdata, sprintf('%s_pq%sinvidx.mat', prms.experiment.codes_suffix, prms.experiment.pq_tag));
kKernelFile = fullfile(prms.paths.compdata, sprintf('%s_%s_K.mat', prms.experiment.name, trainSetStr));
kClassifierFile = fullfile(prms.paths.compdata, sprintf('%s_%s_classifier%s.mat', prms.experiment.name, trainSetStr, prms.experiment.classif_tag));
kResultsFile = fullfile(prms.paths.results, sprintf('%s_%s_pq%sresults%s.mat', prms.experiment.name, testSetStr, prms.experiment.pq_tag, prms.experiment.classif_tag));

% --------------------------------
% Compute Chunks (for all splits)
% --------------------------------
if exist(kChunkIndexFile,'file')
    load(kChunkIndexFile)
else
    chunk_files = featpipem.chunkio.compChunksIMDB(prms, featextr, pooler);
    % save chunk_files to file
    save(kChunkIndexFile, 'chunk_files');
end


train_chunks = cell(1,length(prms.splits.train));
for si = 1:length(prms.splits.train)
    train_chunks{si} = chunk_files(prms.splits.train{si});
end

% --------------------------------
% Compute PQ Codebook
% --------------------------------

if exist(kPQCodebookFile,'file')
    load(kPQCodebookFile);
else
    pqcodebkgen = featpipem.pq.PQCodebkGen(codebook, prms.subquant_count, prms.subquant_bits);
    trainvecs = featpipem.chunkio.loadChunksIntoMat(train_chunks);
    pqcodebook = pqcodebkgen.train(trainvecs);
    save(kPQCodebookFile, 'pqcodebook');
    clear trainvecs pqcodebkgen;
end

% --------------------------------
% Compute PQ Codes (for test set)
% --------------------------------

pqencoder = featpipem.pq.PQEncoder(pqcodebook);

if exist(kPQCodesFile,'file')
    load(kPQCodesFile);
else
    pqcodes = containers.Map();
    for setName = prms.splits.test
        fprintf('Computing PQ codes for set %s...\n',setName{1});
        maxidx = 0;
        chunk_files_set = chunk_files(setName{1});
        for ci = 1:length(chunk_files_set)
            ch = load(chunk_files_set{ci});
            
            if ch.index(end) > maxidx, maxidx = ch.index(end); end
            
            % if this is first chunkfile, preallocate output matrix
            if (ci == 1)
                featcount = size(ch.chunk,2);
                chunkfilecount = length(chunk_files_set);
                pqcodelen = pqencoder.get_output_dim();
                pqcodecls = pqencoder.get_output_class();
                pqcodes_set = cast(zeros(pqcodelen, featcount*chunkfilecount), pqcodecls);
            end
            
            % now compute the pq codes for the current chunk
            chunk_offset = ch.index(1)-1; % compute offset in current chunk
            parfor codeidx = ch.index
                fprintf('Computing PQ code %d...\n',codeidx);
                pqcodes_set(:, codeidx) = pqencoder.encode(ch.chunk(:, codeidx-chunk_offset));
            end
        end
        
        pqcodes(setName{1}) = pqcodes_set;
        clear pqcodes_set;
        
        % finally, downsize the output matrix if required
        if (maxidx < size(pqcodes(setName{1}),2))
            pqcodestmp = pqcodes(setName{1});
            pqcodes(setName{1}) = pqcodestmp(:,1:maxidx);
            clear pqcodestmp;
        end
    end
    
    % save the codes to file
    save(kPQCodesFile, 'pqcodes');
end

% % --------------------------------
% % Compute PQ Index
% % --------------------------------
% 
% if exist(kPQInvIdxFile, 'file')
%     load(kPQInvIdxFile);
% else
%     pqindexer = featpipem.pq.PQIndex(pqcodebook);
%     for i = 1:length(prms.splits.test)
%         fprintf(['Indexing for ' prms.splits.test{i} ' set...\n']);
%         pqindexer.index(pqcodes(prms.splits.test{i}));
%     end
%     
%     save(kPQInvIdxFile, 'pqindexer');
% end

% --------------------------------
% Compute Kernel (if using a dual classifier)
% --------------------------------
if isa(classifier, 'featpipem.classification.svm.LibSvmDual')
    if exist(kKernelFile,'file')
        load(kKernelFile);
    else
        K = featpipem.chunkio.compKernel(train_chunks);
        % save kernel matrix to file
        save(kKernelFile, 'K');
    end
end

% --------------------------------
% Train Classifier (using original full codes)
% --------------------------------
if isa(classifier, 'featpipem.classification.svm.LibSvmDual')
    % ...........................
    % training for svm in dual
    % ...........................
    if exist(kClassifierFile,'file')
        load(kClassifierFile);
        classifier.set_model(model); %#ok<NODEF>
    else
        labels_train = featpipem.utility.getImdbGT(prms.imdb, prms.splits.train, 'concatOutput', true);
        classifier.train(K, labels_train, train_chunks);
        model = classifier.get_model(); %#ok<NASGU>
        save(kClassifierFile,'model');
    end
else
    % ...........................
    % training for svm in primal
    % ...........................
    if exist(kClassifierFile,'file')
        load(kClassifierFile);
        classifier.set_model(model); %#ok<NODEF>
    else
        labels_train = featpipem.utility.getImdbGT(prms.imdb, prms.splits.train, 'concatOutput', true);
        trainvecs = featpipem.chunkio.loadChunksIntoMat(train_chunks);
        classifier.train(trainvecs, labels_train);
        model = classifier.get_model(); %#ok<NASGU>
        save(kClassifierFile,'model');
        clear trainvecs;
    end
end

% --------------------------------
% Test Classifier
% --------------------------------
scoremat = cell(1,length(prms.splits.test));
res = cell(1,length(prms.splits.test));

fprintf('Retrieving w matrix...\n');

% Get SVM W Matrix
wmat = classifier.getWMat();
num_classes = size(wmat,2);

% precompute w{j}.c{jt} LUT of dimension # subquants x subquant clusters
fprintf('Precomputing w{j}.c{jt} lookup table...\n');
wcLUT = cell(1, num_classes); %wcLUT is a cell with #classes elements

tLUTCompTime = tic;
for ci = 1:num_classes % iterate through classes
    startidx = 1;
    idxinc = size(pqcodebook{1},1);
    endidx = idxinc;
    wcLUT{ci} = single(zeros(length(pqcodebook), size(pqcodebook{1},2)));
    for qi = 1:length(pqcodebook) % iterate through subquantizers
        if size(pqcodebook{qi},1) ~= idxinc, error('subquantizers have different dimensions'); end
        for wi = 1:size(pqcodebook{qi},2) % iterate through subquantizer clusters
            wcLUT{ci}(qi, wi) = wmat(startidx:endidx,ci)'*pqcodebook{qi}(:,wi);
        end
        startidx = startidx + idxinc;
        endidx = endidx + idxinc;
    end
    % add bias term to all elements - this is incorrect!
    %wcLUT{ci} = wcLUT{ci} + wmat(end,ci);
end
fprintf('Time to compute LUT: %f seconds\n', toc(tLUTCompTime));


% apply classifier to all testsets in prms.splits.test

fprintf('Applying classifier...\n');

biasterm = single(wmat(end, :));

rankingTime = 0;

for si = 1:length(prms.splits.test)
    pqcodes_set = pqcodes(prms.splits.test{si});
    
    tRankingTime = tic;

    scoremat_pf = featpipem.pq.calcScoremat(wcLUT, biasterm, pqcodes_set);
     
    rankingTime = rankingTime + toc(tRankingTime);
    scoremat{si} = scoremat_pf;
    
    switch prms.experiment.evalusing
        case 'precrec'
            res{si} = featpipem.eval.evalPrecRec(prms.imdb, scoremat{si}, prms.splits.test{si}, prms.experiment.dataset);
        case 'accuracy'
            res{si} = featpipem.eval.evalAccuracy(prms.imdb, scoremat{si}, prms.splits.test{si});
        otherwise
            error('Unknown evaluation method %s', prms.experiment.evalusing);
    end
end

fprintf('Ranking time was: %f seconds\n',rankingTime);

% package results
results.res = res;
results.scoremat = scoremat; %#ok<STRNU>
parameters.prms = prms;
parameters.codebook = codebook;
parameters.pqcodebook = pqcodebook;
parameters.featextr = featextr;
parameters.encoder = encoder;
parameters.pooler = pooler;
parameters.classifier = classifier; %#ok<STRNU>
% save results to file
save(kResultsFile, 'results', 'parameters','-v7.3');
    
end

