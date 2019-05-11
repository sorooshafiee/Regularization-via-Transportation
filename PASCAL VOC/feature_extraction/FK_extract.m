clc
clear all
voc_size = 256; % vocabulary size
desc_dim = 80; % descriptor dimensionality after PCA
DataDir = pwd;

%% initialize experiment parameters
prms.experiment.name = 'FK'; % experiment name - prefixed to all output files other than codes
prms.experiment.codes_suffix = 'FK'; % string prefixed to codefiles (to allow sharing of codes between multiple experiments)
prms.experiment.classif_tag = ''; % additional string added at end of classifier and results files (useful for runs with different classifier parameters)
prms.imdb = load(fullfile(DataDir,'imdb/imdb-VOC2007.mat')); % IMDB file
prms.codebook = fullfile(DataDir, sprintf('FK/codebooks/fkdemo_gmm_%d_%d.mat', voc_size, desc_dim)); % desired location of codebook
prms.dimred = fullfile(DataDir, sprintf('FK/dimred/fkdemo_pca_%d.mat', desc_dim)); % desired location of low-dim projection matrix
prms.experiment.dataset = 'VOC2007'; % dataset name - currently only VOC2007 supported

prms.paths.dataset = '/home/shafieez/data/VOCdevkit'; % path to datasets
prms.paths.codes = fullfile(DataDir,'FK/codes/'); % path where codefiles should be stored

prms.chunkio.chunk_size = 100; % number of encodings to store in single chunk
prms.chunkio.num_workers = 4; % number of workers to use when generating chunks

% initialize split parameters
prms.splits.train = {'train', 'val'}; % cell array of IMDB splits to use when training
prms.splits.test = {'test'}; % cell array of IMDB splits to use when testing

% initialize experiment classes
featextr = featpipem.features.IterDSiftExtractor();

%% train/load dimensionality reduction
if desc_dim ~= 128
    dimred = featpipem.dim_red.PCADimRed(featextr, desc_dim);
    is_aug = featextr.aug_frames;
    featextr.aug_frames = false;
    featextr.low_proj = featpipem.wrapper.loaddimred(dimred, prms);
    featextr.aug_frames = is_aug;
else
    % no dimensionality reduction
    featextr.low_proj = [];
end

%% train/load codebook
codebkgen = featpipem.codebkgen.GMMCodebkGen(featextr, voc_size);
codebkgen.GMM_init = 'kmeans';

codebook = featpipem.wrapper.loadcodebook(codebkgen, prms);

%% initialize encoder + pooler
encoder = featpipem.encoding.FKEncoder(codebook);
encoder.pnorm = single(0.0);
encoder.alpha = single(1.0);
encoder.grad_weights = false;
encoder.grad_means = true;
encoder.grad_variances = true;

pooler = featpipem.pooling.SELDMPooler(encoder);
pooler.subbin_norm_type = 'l2';
pooler.norm_type = 'none';
pooler.kermap = 'hellinger';
pooler.post_norm_type = 'l2';

%% compute features
[train_chunks, val_chunks, test_chunks] = featpipem.wrapper.dstest(prms, featextr, pooler);
kChunkTrainIndexFile = fullfile(prms.paths.codes, sprintf('%s_train_files.mat', prms.experiment.codes_suffix));
kChunkValIndexFile = fullfile(prms.paths.codes, sprintf('%s_val_files.mat', prms.experiment.codes_suffix));
kChunkTestIndexFile = fullfile(prms.paths.codes, sprintf('%s_test_files.mat', prms.experiment.codes_suffix));
save(kChunkTrainIndexFile, 'train_chunks');
save(kChunkValIndexFile, 'val_chunks');
save(kChunkTestIndexFile, 'test_chunks');