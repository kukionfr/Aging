%% make image datastore
path(path,'\\kukibox\research\aging\code\MATLAB dependencies');
Openslide_svstotif(pth,'5x\',2,[],'tmp.mat')
pth = '\\kukibox\research\aging\data\svs\multiclass annotation\michael';
pthim = fullfile(pth,'5x');
pthdata=[pth,'mat_files\'];
pthCNN = 'ClassifiedX';
% 1 path to mat files
if ~isfolder(pthdata);mkdir(pthdata);end
if ~isfolder(pthCNN);mkdir(pthCNN);end
% reads xml annotation files and saves as mat files
load_xml(pth,pthdata,0.5);
% calculate tissue space
pthTA=calculate_tissue_space(pthim);

WS{1}=[0,0,0,0,0,0,0,0,0,0,0,0,0,0];   % remove whitespace if 0, keep only whitespace if 1
WS{2}=1;                       % add removed whitespace to this class
WS{3}=[1,2];   % rename classes accoring to this order
WS{4}=[2,1];  % reverse priority of classes
WS{5}=[];

umpix = 2; %5x

% 3 fill annotation outlines and delete unwanted pixels
fill_annotations(pth,pthim,pthdata,WS,umpix,pthTA);

%% KYU LOOK AT THESE YOU'LL HAVE TO EDIT CODE
% make training tiles
pthtiles=[pth,'view_annotations\'];
make_image_tiles(pthim,pthtiles);

% normalize class number in training tiles
normalize_annotations_per_class

% substitute empty space with white
whitespace_tiles

% make tiles to classify from h&e
make_classification_tiles;

% make classified tiles back into image shape
make_images_from_tiles;
