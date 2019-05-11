clc
clear all

% net_name = 'imagenet-vgg-m'; layer = 19;
net_name = 'imagenet-caffe-alex'; layer = 19;
% net_name = 'imagenet-vgg-verydeep-16'; layer = 36;

% compile MatConvNet
matconvnet_loc = '/home/shafieez/Desktop/matconvnet-1.0-beta25/';
run(fullfile(matconvnet_loc,'matlab/vl_setupnn'))
DataDir = pwd;

%% initialize experiment parameters
prms.experiment.name = net_name;            % experiment name 
prms.experiment.codes_suffix = net_name;    % string prefixed to codefiles
prms.imdb = load(fullfile(DataDir,'imdb/imdb-VOC2007.mat')); % IMDB file

prms.experiment.dataset = 'VOC2007'; % dataset name - currently only VOC2007 supported

prms.paths.dataset = '/home/shafieez/data/VOCdevkit'; % path to datasets
prms.paths.codes = fullfile(DataDir,[net_name, '/codes/']); % path where codefiles should be stored
prms.paths.net = fullfile(DataDir,[net_name, '/net/']); % path where codefiles should be stored

prms.chunkio.chunk_size = 100; % number of encodings to store in single chunk
prms.chunkio.num_workers = 4; % number of workers to use when generating chunks

% initialize split parameters
prms.splits.train = {'train', 'val'}; % cell array of IMDB splits to use when training
prms.splits.test = {'test'}; % cell array of IMDB splits to use when testing


% Download a pre-trained CNN from the web (needed once).
filename = fullfile(prms.paths.net, [net_name, '.mat']);
if ~exist(filename, 'file')
    url = ['http://www.vlfeat.org/matconvnet/models/', [net_name, '.mat']];
    websave(filename, url);
end
% Load a model and upgrade it to MatConvNet current version.
net = load(filename);
net = vl_simplenn_tidy(net) ;

% initialize experiment classes
featextr = featpipem.features.DeepExtractor();
featextr.net = net;
featextr.layer = layer;

pooler = featpipem.pooling.IdentityPooler();
pooler.n_dim = 4096;
pooler.post_norm_type = 'l2';

%% compute features
[train_chunks, val_chunks, test_chunks] = featpipem.wrapper.dstest(prms, featextr, pooler);
kChunkTrainIndexFile = fullfile(prms.paths.codes, sprintf('%s_train_files.mat', prms.experiment.codes_suffix));
kChunkValIndexFile = fullfile(prms.paths.codes, sprintf('%s_val_files.mat', prms.experiment.codes_suffix));
kChunkTestIndexFile = fullfile(prms.paths.codes, sprintf('%s_test_files.mat', prms.experiment.codes_suffix));
save(kChunkTrainIndexFile, 'train_chunks');
save(kChunkValIndexFile, 'val_chunks');
save(kChunkTestIndexFile, 'test_chunks');