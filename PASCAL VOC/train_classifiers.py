""" Implementation of DRO models """

import gurobipy as grb
import numpy as np
from time import time


def train_classifier(Kernel, labels_raw, all_epsilon, all_kappa, nc):
    
    print('Train class ', nc + 1, '...')
    t = time()

    n_samples = Kernel.shape[0]
    alpha = np.zeros((n_samples, len(all_kappa), len(all_epsilon)))

    labels = -np.ones(n_samples)
    labels[labels_raw[nc]] = 1

    for nk, kappa in enumerate(all_kappa):
        for ne, epsilon in enumerate(all_epsilon):
            optimal = ksvm(Kernel, labels, epsilon, kappa)
            alpha[:, nk, ne] = optimal['alpha']
    elapsed = time() - t
    print('Class ', nc + 1, ' is trained in ', np.round(elapsed/60.0, 2), ' minutes.')
    return alpha

def ksvm(Kernel, labels, epsilon, kappa):
    """ kernelized SVM """
    certif = np.linalg.eigvalsh(Kernel)[0]
    if certif < 0:
        Kernel = Kernel - 2 * certif * np.eye(Kernel.shape[0])
    if epsilon == 0:
        optimal = hinge_ksvm(Kernel, labels)
    elif np.isinf(kappa):
        optimal = regularized_ksvm(Kernel, labels, epsilon)
    else:
        optimal = dist_rob_ksvm(Kernel, labels, epsilon, kappa)

    return optimal

def dist_rob_ksvm(Kernel, labels, epsilon, kappa):
    """ kernelized distributionally robust SVM """

    n_samples = Kernel.shape[0]

    # Step 0: create model
    model = grb.Model('Ker_DRSVM')
    model.setParam('OutputFlag', False)

    # Step 1: define decision variables
    var_lambda = model.addVar(vtype=grb.GRB.CONTINUOUS)
    var_s = {}
    var_alpha = {}
    for i in range(n_samples):
        var_s[i] = model.addVar(vtype=grb.GRB.CONTINUOUS)
        var_alpha[i] = model.addVar(
            vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY)

    # Step 2: integerate variables
    model.update()

    # Step 3: define constraints
    chg_cons = {}
    for i in range(n_samples):
        model.addConstr(
            1 - labels[i] * grb.quicksum(var_alpha[k] * Kernel[k, i]
                                          for k in range(n_samples)) <= var_s[i])
        chg_cons[i] = model.addConstr(
            1 + labels[i] * grb.quicksum(var_alpha[k] * Kernel[k, i]
                                          for k in range(n_samples)) -
            kappa * var_lambda <= var_s[i])
    model.addQConstr(
        grb.quicksum(var_alpha[k1] * Kernel[k1, k2] * var_alpha[k2]
                     for k1 in range(n_samples)
                     for k2 in range(n_samples)) <= var_lambda * var_lambda)

    # Step 4: define objective value
    sum_var_s = grb.quicksum(var_s[i] for i in range(n_samples))
    obj = var_lambda * epsilon + 1.0 / n_samples * sum_var_s
    model.setObjective(obj, grb.GRB.MINIMIZE)

    # Step 5: solve the problem
    model.optimize()

    # Step 6: store results
    alpha_opt = np.array([var_alpha[i].x for i in range(n_samples)])
    optimal = {
        'alpha': alpha_opt,
        'objective': model.ObjVal,
        'diagnosis': model.status
    }

    return optimal

def regularized_ksvm(Kernel, labels, epsilon):
    """ kernelized robust/regularized SVM """

    n_samples = Kernel.shape[0]

    # Step 0: create model
    model = grb.Model('Ker_RSVM')
    model.setParam('OutputFlag', False)

    # Step 1: define decision variables
    var_lambda = model.addVar(vtype=grb.GRB.CONTINUOUS)
    var_s = {}
    var_alpha = {}
    for i in range(n_samples):
        var_s[i] = model.addVar(vtype=grb.GRB.CONTINUOUS)
        var_alpha[i] = model.addVar(
            vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY)

    # Step 2: integerate variables
    model.update()

    # Step 3: define constraints
    for i in range(n_samples):
        model.addConstr(
            1 - labels[i] * grb.quicksum(var_alpha[k] * Kernel[k, i]
                                          for k in range(n_samples)) <= var_s[i])
    model.addQConstr(
        grb.quicksum(var_alpha[k1] * Kernel[k1, k2] * var_alpha[k2]
                     for k1 in range(n_samples)
                     for k2 in range(n_samples)) <= var_lambda * var_lambda)

    # Step 4: define objective value
    sum_var_s = grb.quicksum(var_s[i] for i in range(n_samples))
    obj = var_lambda * epsilon + 1.0 / n_samples * sum_var_s
    model.setObjective(obj, grb.GRB.MINIMIZE)

    # Step 5: solve the problem
    model.optimize()

    # Step 6: store results
    alpha_opt = np.array([var_alpha[i].x for i in range(n_samples)])
    optimal = {
        'alpha': alpha_opt,
        'objective': model.ObjVal,
        'diagnosis': model.status
    }

    return optimal

def hinge_ksvm(Kernel, labels):
    """ kernelized hinge loss minimization """

    n_samples = Kernel.shape[0]

    # Step 0: create model
    model = grb.Model('Ker_RSVM')
    model.setParam('OutputFlag', False)

    # Step 1: define decision variables
    var_s = {}
    var_alpha = {}
    for i in range(n_samples):
        var_s[i] = model.addVar(vtype=grb.GRB.CONTINUOUS)
        var_alpha[i] = model.addVar(
            vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY)

    # Step 2: integerate variables
    model.update()

    # Step 3: define constraints
    for i in range(n_samples):
        model.addConstr(
            1 - labels[i] * grb.quicksum(var_alpha[k] * Kernel[k, i]
                                          for k in range(n_samples)) <= var_s[i])

    # Step 4: define objective value
    sum_var_s = grb.quicksum(var_s[i] for i in range(n_samples))
    obj = 1.0 / n_samples * sum_var_s
    model.setObjective(obj, grb.GRB.MINIMIZE)

    # Step 5: solve the problem
    model.optimize()

    # Step 6: store results
    alpha_opt = np.array([var_alpha[i].x for i in range(n_samples)])
    optimal = {
        'alpha': alpha_opt,
        'objective': model.ObjVal,
        'diagnosis': model.status
    }

    return optimal

