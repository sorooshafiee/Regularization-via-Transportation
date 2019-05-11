from __future__ import print_function, division

import time
import os
import random
import copy
from config import DIR_DATA
from utils import compute_AP
import pretrainedmodels.datasets as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from torch.optim import lr_scheduler
from sklearn.utils.extmath import randomized_svd
import pickle

from torchvision import models, transforms

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEVICE = 'cpu'
BATCH_SIZE = 1
NUM_WORKERS = 0
NUM_CLASSES = 20
DIR_SAVE = './results/'
if not os.path.exists(DIR_SAVE):
    os.makedirs(DIR_SAVE)

seed_num = 12345
torch.manual_seed(seed_num)
np.random.seed(seed_num)
random.seed(seed_num)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='PyTorch AlexNet Finetune')
parser.add_argument('--regularization', default='lasso', type=str, help='Regularization method')
parser.add_argument('--coefficient', default=0, type=float, help='Regularization coefficient')
parser.add_argument('--nepochs', default=50, type=int, help='Number of epochs')
parser.add_argument('--rank', default=0, type=int, help='Rank in randomized SVD')
parser.add_argument('--prox', dest='prox', action='store_true')
parser.add_argument('--no-prox', dest='prox', action='store_false')
parser.set_defaults(prox=True)
args = parser.parse_args()

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def save_obj(obj, fname):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def euclidean_proj_simplex(v, s):
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w


def euclidean_proj_l1ball(v, s):
    if s == 0:
        return np.zeros(v.shape)
    # compute the vector of absolute values
    u = np.abs(v)
    # check if v is already a solution
    if u.sum() <= s:
        # L1-norm is <= s
        return v
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    w = euclidean_proj_simplex(u, s)
    # compute the solution to the original problem on v
    w *= np.sign(v)
    return w


def func(b, Y, coeff):
    obj = coeff * b
    X = np.zeros(Y.shape)
    for i in range(Y.shape[0]):
        w = euclidean_proj_l1ball(Y[i,:], b)
        val = np.linalg.norm(w - Y[i,:])
        obj += 0.5 * val**2
        X[i,:] = w
    return X, obj


def golden_section(Y, coeff, norm):
    phi = (1 + np.sqrt(5)) / 2
    tau = 2 - phi
    if norm == 1:
        Y = Y.T
    a = min(np.abs(Y).sum(axis=1))
    d = np.linalg.norm(Y, np.inf)
    c = (1 - tau) * d + tau * a
    b = (1 - tau) * a + tau * d
    fc = func(c, Y, coeff)[1]
    fb = func(b, Y, coeff)[1]
    for i in range(10):
        if fb <= fc:
            d = c
            c = b
            fc = fb
            b = (1 - tau) * a + tau * d
            fb = func(b, Y, coeff)[1]
        else:
            a = b
            b = c
            fb = fc
            c = (1 - tau) * d + tau * a
            fc = func(c, Y, coeff)[1]
    sol = (a + d) / 2
    X_sol = func(sol, Y, coeff)[0]
    if norm == 1:
        X_sol = X_sol.T
    return torch.from_numpy(X_sol)


def projection(Y, coeff, norm):
    if norm == 1:
        Y = Y.T
    X_sol = np.zeros(Y.shape)
    for i in range(Y.shape[0]):
        w = euclidean_proj_l1ball(Y[i, :], coeff)
        X_sol[i, :] = w
    if norm == 1:
        X_sol = X_sol.T
    return torch.from_numpy(X_sol)


def prox_op(model, lr):
    if args.coefficient == 0:
        return model
    with torch.no_grad():
        for p in model.parameters():

            if p.requires_grad and args.regularization.lower() == 'lasso' and len(p.size()) == 2:
                # elementwise soft thresholding
                p.sub_(p.sign() * p.abs().clamp(max=lr * args.coefficient))

            if p.requires_grad and args.regularization.lower() == 'tikhonov' and len(p.size()) == 2:
                # it is already considered in weight_decay parameter
                pass

            if p.requires_grad and args.regularization.lower() == 'l2' and len(p.size()) == 2:
                # elementwise scaling
                val = 1 - lr * args.coefficient / p.pow(2).sum().sqrt()
                val.clamp_(min=0)
                p.mul_(val)

            if p.requires_grad and args.regularization.lower() == 'mars' and len(p.size()) == 2:
                # row-wise soft thresholding = ell_infty norm
                arg_prox = golden_section(p.data.cpu().numpy(), lr * args.coefficient, np.inf)
                p.data.copy_(arg_prox.data)

            if p.requires_grad and args.regularization.lower() == 'macs' and len(p.size()) == 2:
                # column-wise soft thresholding = ell_1 norm
                arg_prox = golden_section(p.data.cpu().numpy(), lr * args.coefficient, 1)
                p.data.copy_(arg_prox.data)

            if p.requires_grad and args.regularization.lower() == 'spectral' and len(p.size()) == 2:
                # singular value soft thresholding
                n_components = args.rank if args.rank > 0 else min(p.shape)
                U, Sigma, VT = randomized_svd(p.data.cpu().numpy(), n_components=n_components, n_iter=1)
                smat = np.diag(np.maximum(Sigma - lr * args.coefficient, 0))
                arg_prox = torch.from_numpy(U.dot(smat).dot(VT))
                p.data.copy_(arg_prox.data)

    return model


def proj_op(model):
    if args.coefficient == 0:
        return model
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad and args.regularization.lower() == 'mars' and len(p.size()) == 2:
                # row-wise soft thresholding = ell_infty norm
                arg_proj = projection(p.data.cpu().numpy(), args.coefficient, np.inf)
                p.data.copy_(arg_proj.data)
            if p.requires_grad and args.regularization.lower() == 'macs' and len(p.size()) == 2:
                # column-wise soft thresholding = ell_1 norm
                arg_proj = projection(p.data.cpu().numpy(), args.coefficient, 1)
                p.data.copy_(arg_proj.data)
            if p.requires_grad and args.regularization.lower() == 'spectral' and len(p.size()) == 2:
                # singular value clipping
                n_components = args.rank if args.rank > 0 else min(p.shape)
                U, Sigma, VT = randomized_svd(p.data.cpu().numpy(), n_components=n_components, n_iter=1)
                smat = np.diag(np.minimum(Sigma, args.coefficient))
                arg_proj = torch.from_numpy(U.dot(smat).dot(VT))
                p.data.copy_(arg_proj.data)
    return model


def train(model, criterion, loader, optimizer, scheduler):
    scheduler.step()
    model.train()

    running_loss = 0.0
    num_sample = 0
    # Iterate over data.
    for batch_id, batch in enumerate(loader):
        inputs = batch[0]
        labels = batch[2]
        labels[labels == -1] = 0
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if args.prox:
                model = prox_op(model, optimizer.param_groups[-1]['lr'])
            else:
                model = proj_op(model)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        num_sample += inputs.size(0)

    epoch_loss = running_loss / num_sample

    return model, epoch_loss


def validate(model, criterion, loader):
    model.eval()
    running_loss = 0.0
    running_score = torch.zeros(len(loader) * loader.batch_size, NUM_CLASSES)
    targets = torch.zeros(len(loader) * loader.batch_size, NUM_CLASSES)

    # Iterate over data
    to_ = 0
    with torch.no_grad():
        for batch_id, batch in enumerate(loader):
            inputs = batch[0]
            target = batch[2]
            target[target == -1] = 0
            inputs = inputs.to(DEVICE)
            target = target.to(DEVICE)

            current_bsize = inputs.size(0)
            from_ = int(batch_id * loader.batch_size)
            to_ = int(from_ + current_bsize)

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, target)
            running_loss += loss.item() * inputs.size(0)
            running_score[from_:to_] = outputs
            targets[from_:to_] = target

    running_score = running_score[:to_, :]
    targets = targets[:to_, :]
    epoch_AP = compute_AP(running_score.data.cpu().numpy(), targets.data.cpu().numpy())
    epoch_loss = running_loss / to_

    return epoch_AP, epoch_loss


def train_model(model, criterion, dataloaders, optimizer, scheduler, num_epochs=100):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_mAP = 0
    AP_val = 0
    loss_val = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                t_train = time.time()
                model, epoch_loss = train(model, criterion, dataloaders[phase], optimizer, scheduler)
                t_train = time.time() - t_train
                print('Training epoch in {:.0f}m {:.0f}s => train loss: {:.4f}'.format(
                    t_train // 60, t_train % 60, epoch_loss))
            else:
                epoch_AP, epoch_loss = validate(model, criterion, dataloaders[phase])
                print('Validation loss: {:.4f} mAP: {:.4f}'.format(epoch_loss, np.mean(epoch_AP)))
                # deep copy the model
                if np.mean(epoch_AP) > best_mAP:
                    print('A new model is saved in epoch ', epoch)
                    AP_val = epoch_AP
                    loss_val = epoch_loss
                    best_mAP = np.mean(epoch_AP)
                    best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, AP_val, loss_val


def main():

    if args.regularization.lower() in ['tikhonov', 'lasso', 'l2'] and not args.prox:
        print('Projection operator is not supported for {}. Change it to proximal operator.'.format(
            args.regularization.lower()))
        return

    # Print the objective of training
    print('=' * 20)
    print('Training {} regularization with coefficient {:.1e}'.format(
        args.regularization.lower(), args.coefficient))
    print('-' * 20)
    results = {}

    # Create net name
    reg_name = args.regularization.lower() + '_' + str(args.coefficient) + '_' if args.coefficient != 0 else ''
    dict_name = reg_name + 'results.pkl'
    file_path = os.path.join(DIR_SAVE, dict_name)

    if os.path.isfile(file_path):
        print('Deep network was already trained')
        print('Loading the results ...')
        results = load_obj(file_path)
        AP_test, loss_test = results['test']
        AP_val, loss_val = results['val']
    else:
        # Load AlexNet
        model_ft = models.alexnet(pretrained=True)
        for param in model_ft.parameters():
            param.requires_grad = False
        num_ftrs = model_ft.classifier[-1].in_features
        model_ft.classifier[-1] = nn.Linear(num_ftrs, NUM_CLASSES)
        model_ft = model_ft.to(DEVICE)

        # Loss function to minimize
        loss_func = nn.BCEWithLogitsLoss()

        # Add L2 penalty term if we regularize the model with Tikhonov method
        weight_decay = args.coefficient if args.regularization.lower() == 'tikhonov' else 0

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9, weight_decay=weight_decay)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        # Load data
        image_datasets = {x: pd.Voc2007Classification(DIR_DATA, x,
                                                      transform=data_transforms[x])
                          for x in ['train', 'val', 'test']}

        shuffle = {'train': True, 'val': False, 'test': False}

        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                                      shuffle=shuffle[x], num_workers=NUM_WORKERS)
                       for x in ['train', 'val', 'test']}

        # Train model
        model_ft, AP_val, loss_val = train_model(
            model_ft, loss_func, dataloaders, optimizer_ft, exp_lr_scheduler, num_epochs=args.nepochs)

        # Save test results
        AP_test, loss_test = validate(model_ft, loss_func, dataloaders['test'])
        results['test'] = AP_test, loss_test
        results['val'] = AP_val, loss_val
        save_obj(results, file_path)

    print('Validatin loss: {:.4f} mAP: {:.4f}'.format(loss_val, np.mean(AP_val)))
    print('Test loss: {:.4f} mAP: {:.4f}'.format(loss_test, np.mean(AP_test)))
    print(AP_test)
    print('=' * 20)


if __name__ == '__main__':
    main()
