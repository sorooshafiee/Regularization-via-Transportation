#!/bin/sh
# Without regularization
python finetune.py
# Spectral regularization
python finetune.py --regularization spectral --coefficient 0.30 --no-prox
python finetune.py --regularization spectral --coefficient 0.31 --no-prox
python finetune.py --regularization spectral --coefficient 0.32 --no-prox
python finetune.py --regularization spectral --coefficient 0.33 --no-prox
python finetune.py --regularization spectral --coefficient 0.34 --no-prox
python finetune.py --regularization spectral --coefficient 0.35 --no-prox
python finetune.py --regularization spectral --coefficient 0.36 --no-prox
python finetune.py --regularization spectral --coefficient 0.37 --no-prox
python finetune.py --regularization spectral --coefficient 0.38 --no-prox
python finetune.py --regularization spectral --coefficient 0.39 --no-prox
python finetune.py --regularization spectral --coefficient 0.40 --no-prox
# MACS regularization
python finetune.py --regularization macs --coefficient 0.10 --no-prox
python finetune.py --regularization macs --coefficient 0.11 --no-prox
python finetune.py --regularization macs --coefficient 0.12 --no-prox
python finetune.py --regularization macs --coefficient 0.13 --no-prox
python finetune.py --regularization macs --coefficient 0.14 --no-prox
python finetune.py --regularization macs --coefficient 0.15 --no-prox
python finetune.py --regularization macs --coefficient 0.16 --no-prox
python finetune.py --regularization macs --coefficient 0.17 --no-prox
python finetune.py --regularization macs --coefficient 0.18 --no-prox
python finetune.py --regularization macs --coefficient 0.19 --no-prox
python finetune.py --regularization macs --coefficient 0.20 --no-prox
# MARS regularization
python finetune.py --regularization mars --coefficient 10 --no-prox
python finetune.py --regularization mars --coefficient 20 --no-prox
python finetune.py --regularization mars --coefficient 30 --no-prox
python finetune.py --regularization mars --coefficient 40 --no-prox
python finetune.py --regularization mars --coefficient 50 --no-prox
# Lasso regularization
python finetune.py --regularization lasso --coefficient 1e-4
python finetune.py --regularization lasso --coefficient 2e-4
python finetune.py --regularization lasso --coefficient 3e-4
python finetune.py --regularization lasso --coefficient 4e-4
python finetune.py --regularization lasso --coefficient 5e-4
python finetune.py --regularization lasso --coefficient 6e-4
python finetune.py --regularization lasso --coefficient 7e-4
python finetune.py --regularization lasso --coefficient 8e-4
python finetune.py --regularization lasso --coefficient 9e-4
python finetune.py --regularization lasso --coefficient 1e-3
# Tikhonov Regularization
python finetune.py --regularization tikhonov --coefficient 1e-3
python finetune.py --regularization tikhonov --coefficient 2e-3
python finetune.py --regularization tikhonov --coefficient 3e-3
python finetune.py --regularization tikhonov --coefficient 4e-3
python finetune.py --regularization tikhonov --coefficient 5e-3
python finetune.py --regularization tikhonov --coefficient 6e-3
python finetune.py --regularization tikhonov --coefficient 7e-3
python finetune.py --regularization tikhonov --coefficient 8e-3
python finetune.py --regularization tikhonov --coefficient 9e-3
python finetune.py --regularization tikhonov --coefficient 1e-2
