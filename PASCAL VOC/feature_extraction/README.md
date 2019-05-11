Encoding Methods Evaluation Toolkit
============================================

Author: Ken Chatfield, University of Oxford (ken@robots.ox.ac.uk)

Copyright 2011-2013, all rights reserved.

Release: v1.2

Licence: BSD (see COPYING file)
-----------------------------

The script FK_extract.m extracts the Fisher kernel encoding from SIFT
descriptors. In order to run the code, first open the script and modify
the prms.paths.dataset variable to point to the directory in which you
have saved the PASCAL VOC 2007 dataset.

This demo performs the following steps:

1. Trains a vocabulary of size 4,000 visual words from SIFT features
   densely extracted from the PASCAL VOC 2007 train/val set
2. Computes the stanard hard assignment bag-of-words encoding for all
   images in the PASCAL VOC 2007 dataset using this vocabulary
3. Applies an additive kernel map (hellinger) to all codes

NOTE: The code used to train GMM vocabularies and compute the Fisher kernel
encoding is provided by the 'GMM-Fisher' sublibrary, found in the
`lib/gmm-fisher/` subdirectory. Unfortunatly, Windows is not supported at this stage.