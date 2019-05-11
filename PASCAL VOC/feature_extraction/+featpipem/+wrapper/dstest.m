function [train_chunks, val_chunks, test_chunks] = dstest(prms, featextr, pooler)
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


% --------------------------------
% Compute Chunks (for all splits)
% --------------------------------

chunk_files = featpipem.chunkio.compChunksIMDB(prms, featextr, pooler);
% save chunk_files to file
save(kChunkIndexFile, 'chunk_files');

train_chunks = chunk_files(prms.splits.train{1});
val_chunks = chunk_files(prms.splits.train{2});
test_chunks = chunk_files(prms.splits.test{1});