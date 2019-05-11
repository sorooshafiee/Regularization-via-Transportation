from scipy.io import loadmat
from config import DIR_IMDB
from collections import namedtuple
import numpy as np

data = loadmat(DIR_IMDB + 'imdb-VOC2007.mat')
###
IMDB = namedtuple('IMDB', ['classes', 'images', 'sets'])
CLASSES = namedtuple('CLASSES', ['name', 'imageIds'])
IMAGES = namedtuple('IMAGES', ['id', 'set', 'name', 'size'])
SETS = namedtuple('SETS', ['TRAIN', 'VAL', 'TEST'])
###
name = np.concatenate(data['classes'][0][0][0][0], axis=0)
imageIds = data['classes'][0][0][1][0]
for count, imageId in enumerate(imageIds):
    imageIds[count] = np.squeeze(imageId)
classes = CLASSES(name, imageIds)
###
ids = np.squeeze(data['images'][0][0][0])
sets = np.squeeze(data['images'][0][0][1])
names = np.concatenate(np.squeeze(data['images'][0][0][2]), axis=0)
sizes = np.squeeze(data['images'][0][0][3])
images = IMAGES(ids, sets, names, sizes)
###
sets = SETS(1, 2, 3)
###
imdb = IMDB(classes, images, sets)