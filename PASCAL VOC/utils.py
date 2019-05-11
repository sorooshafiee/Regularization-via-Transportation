import os
import h5py
import numpy as np
from shutil import copy2
from config import DATASET, imdb
from os.path import isfile
from train_classifiers import train_classifier


def _load_mat(chunk_file):
    with h5py.File(chunk_file, 'r') as f:
        chunk = f['chunk'][()].T
        index = f['index'][()].ravel().astype(int) - 1
    return chunk, index

def _voc_ap(rec, prec):
    ap = 0
    for t in np.arange(0,1.1,0.1):
        try:
            p = max(prec[rec>=t])
        except:
            p = 0
        ap += p/11
    return ap

def _n_classes():
    return imdb.classes.name.shape[-1]

def _set_sizes(sets=DATASET):
    return [sum(imdb.images.set == getattr(imdb.sets, ch.upper())) for ch in sets]

def _index_of_class(data, ci):
    setid = getattr(imdb.sets, data.upper())
    image_ids = imdb.images.id[imdb.images.set == setid]
    return np.where(np.in1d(image_ids, imdb.classes.imageIds[ci]))[0]

def _get_id(sets):
    n_classes = _n_classes()
    set_sizes = _set_sizes()

    gt = [np.array([], dtype=np.int64)] * n_classes

    for ci in range(n_classes):

        for si, data in enumerate(DATASET):

            if data not in sets:
                continue

            ind = _index_of_class(data, ci) + sum(set_sizes[:si])
            gt[ci] = np.append(gt[ci], ind)

    return gt
    
def get_labels(sets):
    n_classes = _n_classes()
    set_sizes = _set_sizes(sets)

    gt = [np.array([], dtype=np.int64)] * n_classes

    for ci in range(n_classes):

        for si, data in enumerate(sets):

            ind = _index_of_class(data, ci) + sum(set_sizes[:si])
            gt[ci] = np.append(gt[ci], ind)

    return gt

def compute_kernel(chunk_files):

    size_chunk = 100

    size_est = 0
    for data in DATASET:
        size_est = size_est + size_chunk * chunk_files[data].shape[-1]

    K = np.zeros([size_est, size_est])

    idxoffseti = 0
    maxidx_ker = 0

    for key_i, value_i in chunk_files.items():

        if key_i not in DATASET:
            continue

        idx1 = [None] * value_i.shape[-1]
        K1 = [None] * value_i.shape[-1]

        for ci, file_i in enumerate(value_i):

            # load chunk
            ch1, ind1 = _load_mat(file_i)

            # apply index offset if required
            idx1[ci] = ind1 + idxoffseti

            # part of K
            K1[ci] = np.zeros([ch1.shape[1], size_est])

            # iterate over second chunkfile
            idxoffsetj = 0

            for key_j, value_j in chunk_files.items():

                if key_j not in DATASET:
                    continue

                idx2end = np.zeros(value_j.shape[0])

                for cj, file_j in enumerate(value_j):

                    ch2, ind2 = _load_mat(file_j)

                    # apply index offset if required
                    ind2 = ind2 + idxoffsetj
                    idx2end[cj] = ind2[-1]

                    # do computation of sub-part of kernel matrix
                    K1[ci][:, ind2] = ch1.T.dot(ch2)

                # store maxidxj for current set
                idxoffsetj = np.max(idx2end).astype(int) + 1

        # copy the data from K1 to K
        for ci in range(value_i.shape[-1]):
            K[idx1[ci], :] = K1[ci]

        # store maxidxi for current set
        max_idx1 = np.array([idx1[i][-1]
                             for i in range(value_i.shape[-1])])
        idxoffseti = np.max(max_idx1).astype(int) + 1

        # store absolute max index to aid resizing of kernel matrix
        maxidx_ker = np.maximum(maxidx_ker, idxoffseti).astype(int)

    return K[:maxidx_ker, :maxidx_ker]

def compute_precrec(scores, labels):

    test_size, n_classes = scores.shape
    AP = np.zeros(n_classes)

    gt = -np.ones((test_size, n_classes))
    for nc in range(n_classes):
        
        gt[labels[nc], nc] = 1

    for ci in range(n_classes):
        gt_cls = gt[:,ci]
        
        # compute precision/recall
        sortidx = np.argsort(scores[:,ci])
        sortidx = sortidx[::-1]
        tp = gt_cls[sortidx]>0
        fp = gt_cls[sortidx]<0

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp/sum(gt_cls>0)
        prec = tp/(fp+tp)
        
        AP[ci] = _voc_ap(rec, prec)
    
    return AP

def VOC_to_IMAGENET(dir_voc, dir_imnet='./voc/'):

    if not os.path.exists(dir_imnet):
        os.makedirs(dir_imnet)
    
    for dirname in DATASET:

        dirname = dirname.lower()

        if not os.path.exists(os.path.join(dir_imnet, dirname)):
            os.makedirs(os.path.join(dir_imnet, dirname))

        ind_raw = _get_id(dirname)

        for ci in range(20):
            
            name = imdb.classes.name[ci]
            
            if not os.path.exists(os.path.join(dir_imnet, dirname, name)):
                os.makedirs(os.path.join(dir_imnet, dirname, name))
                
            for ind in ind_raw[ci]:
                copy2(os.path.join(dir_voc, imdb.images.name[ind]),
                      os.path.join(dir_imnet, dirname, name))

