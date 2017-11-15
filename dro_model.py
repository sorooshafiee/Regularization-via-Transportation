""" Implementation of DRO models """

import gurobipy as grb
import numpy as np


def svm(param, data):
    """ Support Vector Machine """
    optimal = {}
    if len(param['d']) == 0:
        if len(param['kappa']) > 1 or float('inf') not in param['kappa']:
            optimal.update(dist_rob_svm_with_support(param, data))
        if float('Inf') in param['kappa']:
            optimal.update(regularized_svm_with_support(param, data))
    else:
        if len(param['kappa']) > 1 or float('inf') not in param['kappa']:
            optimal.update(dist_rob_svm_without_support(param, data))
        if float('Inf') in param['kappa']:
            optimal.update(regularized_svm_without_support(param, data))

    return optimal


def dist_rob_svm_with_support(param, data):
    """ distributionally robust SVM with support information """
    x_train = data['x']
    y_train = data['y'].flatten()

    all_epsilon = list(param['epsilon'])
    all_epsilon.sort(reverse=True)
    all_kappa = list(param['kappa'])
    all_kappa.sort(reverse=True)
    if float('Inf') in all_kappa:
        all_kappa.remove(float('Inf'))
    pnorm = param['pnorm']
    mat_c = param['C']
    vec_d = param['d']

    row, col = x_train.shape
    optimal = {}

    # Step 0: create model
    model = grb.Model('DRSVM_with_support')
    model.setParam('OutputFlag', False)

    # Step 1: define decision variables
    var_lambda = model.addVar(vtype=grb.GRB.CONTINUOUS)
    var_s = {}
    var_w = {}
    for i in range(row):
        var_s[i] = model.addVar(vtype=grb.GRB.CONTINUOUS,)
    for j in range(col):
        var_w[j] = model.addVar(vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY)
    var_p_plus = {}
    var_p_minus = {}
    for i in range(row):
        for k in range(len(vec_d)):
            var_p_plus[i, k] = model.addVar(vtype=grb.GRB.CONTINUOUS)
            var_p_minus[i, k] = model.addVar(vtype=grb.GRB.CONTINUOUS)
    if pnorm == 1:
        slack_var_1 = {}
        slack_var_2 = {}
        for i in range(row):
            for j in range(col):
                slack_var_1[i][j] = model.addVar(vtype=grb.GRB.CONTINUOUS)
                slack_var_2[i][j] = model.addVar(vtype=grb.GRB.CONTINUOUS)
    elif pnorm == float('inf'):
        slack_var_1 = {}
        slack_var_2 = {}
        for i in range(row):
            slack_var_1[i] = model.addVar(vtype=grb.GRB.CONTINUOUS)
            slack_var_2[i] = model.addVar(vtype=grb.GRB.CONTINUOUS)

    # Step 2: integerate variables
    model.update()

    # Step 3: define constraints
    chg_cons = {}
    for i in range(row):
        model.addConstr(
            1 - y_train[i] * grb.quicksum(var_w[j] * x_train[i, j]
                                          for j in range(col)) +
            grb.quicksum(
                var_p_plus[i, k] * (vec_d[k] - mat_c[k,j] * x_train[i, j])
                for j in range(col)
                for k in range(len(vec_d))) <= var_s[i])

        chg_cons[i] = model.addConstr(
            1 + y_train[i] * grb.quicksum(var_w[j] * x_train[i, j]
                                          for j in range(col)) +
            grb.quicksum(
                var_p_minus[i][k] * (vec_d[k] - mat_c[k,j] * x_train[i, j])
                for j in range(col)
                for k in range(len(vec_d))) -
            all_kappa[0] * var_lambda <= var_s[i])
        if pnorm == 1:
            for j in range(col):
                model.addConstr(
                    grb.quicksum(var_p_plus[i, k] * mat_c[k,j]
                                 for k in range(len(vec_d))) +
                    y_train[i] * var_w[j] <= slack_var_1[i][j])
                model.addConstr(
                    grb.quicksum(-var_p_plus[i, k] * mat_c[k,j]
                                 for k in range(len(vec_d))) -
                    y_train[i] * var_w[j] <= slack_var_1[i][j])
                model.addConstr(
                    grb.quicksum(var_p_minus[i][k] * mat_c[k,j]
                                 for k in range(len(vec_d))) -
                    y_train[i] * var_w[j] <= slack_var_2[i][j])
                model.addConstr(
                    grb.quicksum(-var_p_minus[i][k] * mat_c[k,j]
                                 for k in range(len(vec_d))) +
                    y_train[i] * var_w[j] <= slack_var_2[i][j])

            model.addConstr(grb.quicksum(slack_var_1[i][j]
                                         for j in range(col)) <= var_lambda)
            model.addConstr(grb.quicksum(slack_var_2[i][j]
                                         for j in range(col)) <= var_lambda)
        elif pnorm == 2:
            model.addQConstr(
                grb.quicksum(var_w[j] * var_w[j]
                             for j in range(col)) +
                grb.quicksum(
                    var_p_plus[i, k] * mat_c[k,j] * (
                        var_p_plus[i, k] * mat_c[k,j] +
                        2 * y_train[i] * var_w[j])
                    for j in range(col)
                    for k in range(len(vec_d))) <= var_lambda * var_lambda)
            model.addQConstr(
                grb.quicksum(var_w[j] * var_w[j]
                             for j in range(col)) +
                grb.quicksum(
                    var_p_minus[i][k] * mat_c[k,j] * (
                        var_p_minus[i][k] * mat_c[k,j] -
                        2 * y_train[i] * var_w[j])
                    for j in range(col)
                    for k in range(len(vec_d))) <= var_lambda * var_lambda)
        elif pnorm == float('Inf'):
            for j in range(col):
                model.addConstr(
                    grb.quicksum(var_p_plus[i, k] * mat_c[k,j]
                                 for k in range(len(vec_d))) +
                    y_train[i] * var_w[j] <= var_lambda)
                model.addConstr(
                    grb.quicksum(-var_p_plus[i, k] * mat_c[k,j]
                                 for k in range(len(vec_d))) -
                    y_train[i] * var_w[j] <= var_lambda)
                model.addConstr(
                    grb.quicksum(var_p_minus[i][k] * mat_c[k,j]
                                 for k in range(len(vec_d))) -
                    y_train[i] * var_w[j] <= var_lambda)
                model.addConstr(
                    grb.quicksum(-var_p_minus[i][k] * mat_c[k,j]
                                 for k in range(len(vec_d))) +
                    y_train[i] * var_w[j] <= var_lambda)

    # Step 4: define objective value
    sum_var_s = grb.quicksum(var_s[i] for i in range(row))
    for index, kappa in enumerate(all_kappa):
        # Change model for different kappa
        if index > 0:
            for i in range(row):
                model.chgCoeff(chg_cons[i], var_lambda, -kappa)
        for epsilon in all_epsilon:
            obj = var_lambda * epsilon + 1 / row * sum_var_s
            model.setObjective(obj, grb.GRB.MINIMIZE)

            # Step 5: solve the problem
            model.optimize()

            # Step 6: store results
            w_opt = np.array([var_w[i].x for i in range(col)])
            tmp = {
                (kappa, epsilon): {
                    'w': w_opt,
                    'objective': model.ObjVal,
                    'diagnosis': model.status
                }
            }
            optimal.update(tmp)

    return optimal


def dist_rob_svm_without_support(param, data):
    """ distributionally robust SVM without support information """
    x_train = data['x']
    y_train = data['y'].flatten()

    all_epsilon = list(param['epsilon'])
    all_epsilon.sort(reverse=True)
    all_kappa = list(param['kappa'])
    all_kappa.sort(reverse=True)
    if float('Inf') in all_kappa:
        all_kappa.remove(float('Inf'))
    pnorm = param['pnorm']

    row, col = x_train.shape
    optimal = {}

    # Step 0: create model
    model = grb.Model('DRSVM_without_support')
    model.setParam('OutputFlag', False)

    # Step 1: define decision variables
    var_lambda = model.addVar(vtype=grb.GRB.CONTINUOUS)
    var_s = {}
    var_w = {}
    slack_var = {}
    for i in range(row):
        var_s[i] = model.addVar(vtype=grb.GRB.CONTINUOUS,)
    for j in range(col):
        var_w[j] = model.addVar(
            vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY)
        if pnorm == 1:
            slack_var[j] = model.addVar(vtype=grb.GRB.CONTINUOUS)

    # Step 2: integerate variables
    model.update()

    # Step 3: define constraints
    chg_cons = {}
    for i in range(row):
        model.addConstr(
            1 - y_train[i] * grb.quicksum(var_w[j] * x_train[i, j]
                                          for j in range(col)) <= var_s[i])
        chg_cons[i] = model.addConstr(
            1 + y_train[i] * grb.quicksum(var_w[j] * x_train[i, j]
                                          for j in range(col)) -
            all_kappa[0] * var_lambda <= var_s[i])

    if pnorm == 1:
        for j in range(col):
            model.addConstr(var_w[j] <= slack_var[j])
            model.addConstr(-var_w[j] <= slack_var[j])
        model.addConstr(grb.quicksum(slack_var[j]
                                     for j in range(col)) <= var_lambda)
    elif pnorm == 2:
        model.addQConstr(
            grb.quicksum(var_w[j] * var_w[j]
                         for j in range(col)) <= var_lambda * var_lambda)

    elif pnorm == float('Inf'):
        for j in range(col):
            model.addConstr(var_w[j] <= var_lambda)
            model.addConstr(-var_w[j] <= var_lambda)

    # Step 4: define objective value
    sum_var_s = grb.quicksum(var_s[i] for i in range(row))
    for index, kappa in enumerate(all_kappa):
        # Change model for different kappa
        if index > 0:
            for i in range(row):
                model.chgCoeff(chg_cons[i], var_lambda, -kappa)
        for epsilon in all_epsilon:
            obj = var_lambda * epsilon + 1 / row * sum_var_s
            model.setObjective(obj, grb.GRB.MINIMIZE)

            # Step 5: solve the problem
            model.optimize()

            # Step 6: store results
            w_opt = np.array([var_w[i].x for i in range(col)])
            tmp = {
                (kappa, epsilon): {
                    'w': w_opt,
                    'objective': model.ObjVal,
                    'diagnosis': model.status
                }
            }
            optimal.update(tmp)

    return optimal


def regularized_svm_with_support(param, data):
    """ regularized/robust SVM with support information """
    x_train = data['x']
    y_train = data['y'].flatten()

    all_epsilon = list(param['epsilon'])
    all_epsilon.sort(reverse=True)
    pnorm = param['pnorm']
    mat_c = param['C']
    vec_d = param['d']

    row, col = x_train.shape
    optimal = {}

    # Step 0: create model
    model = grb.Model('RSVM_with_support')
    model.setParam('OutputFlag', False)

    # Step 1: define decision variables
    var_lambda = model.addVar(vtype=grb.GRB.CONTINUOUS)
    var_s = {}
    var_w = {}
    var_p_plus = {}
    slack_var = {}
    for i in range(row):
        var_s[i] = model.addVar(vtype=grb.GRB.CONTINUOUS)
        for k in range(len(vec_d)):
            var_p_plus[i, k] = model.addVar(vtype=grb.GRB.CONTINUOUS)
    for j in range(col):
        var_w[j] = model.addVar(
            vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY)
        if pnorm == 1:
            for i in range(row):
                slack_var[i, j] = model.addVar(vtype=grb.GRB.CONTINUOUS)

    # Step 2: integerate variables
    model.update()

    # Step 3: define constraints
    for i in range(row):
        model.addConstr(
            1 - y_train[i] * grb.quicksum(var_w[j] * x_train[i, j]
                                          for j in range(col)) <= var_s[i])
        if pnorm == 1:
            for j in range(col):
                model.addConstr(
                    grb.quicksum(var_p_plus[i, k] * mat_c[k,j]
                                 for k in range(len(vec_d))) +
                    y_train[i] * var_w[j] <= slack_var[i, j])
                model.addConstr(
                    grb.quicksum(-var_p_plus[i, k] * mat_c[k,j]
                                 for k in range(len(vec_d))) -
                    y_train[i] * var_w[j] <= slack_var[i, j])

            model.addConstr(grb.quicksum(slack_var[i, j]
                                         for j in range(col)) <= var_lambda)
        elif pnorm == 2:
            model.addQConstr(grb.quicksum(var_w[j] * var_w[j]
                                          for j in range(col)) +
                             grb.quicksum(
                                 var_p_plus[i, k] * mat_c[k,j] * (
                                     var_p_plus[i, k] * mat_c[k,j] +
                                     2 * y_train[i] * var_w[j])
                                 for j in range(col)
                                 for k in range(len(vec_d))) <=
                             var_lambda * var_lambda)
        elif pnorm == float('Inf'):
            for j in range(col):
                model.addConstr(
                    grb.quicksum(var_p_plus[i, k] * mat_c[k,j]
                                 for k in range(len(vec_d))) +
                    y_train[i] * var_w[j] <= var_lambda)
                model.addConstr(
                    grb.quicksum(-var_p_plus[i, k] * mat_c[k,j]
                                 for k in range(len(vec_d))) -
                    y_train[i] * var_w[j] <= var_lambda)

    # Step 4: define objective value
    sum_var_s = grb.quicksum(var_s[i] for i in range(row))
    for epsilon in all_epsilon:
        obj = var_lambda * epsilon + 1 / row * sum_var_s
        model.setObjective(obj, grb.GRB.MINIMIZE)

        # Step 5: solve the problem
        model.optimize()

        # Step 6: store results
        w_opt = np.array([var_w[i].x for i in range(col)])
        tmp = {
            (float('Inf'), epsilon): {
                'w': w_opt,
                'objective': model.ObjVal,
                'diagnosis': model.status
            }
        }
        optimal.update(tmp)

    return optimal


def regularized_svm_without_support(param, data):
    """ regularized/robust SVM without support information """
    x_train = data['x']
    y_train = data['y'].flatten()

    all_epsilon = list(param['epsilon'])
    all_epsilon.sort(reverse=True)
    pnorm = param['pnorm']

    row, col = x_train.shape
    optimal = {}

    # Step 0: create model
    model = grb.Model('RSVM_without_support')
    model.setParam('OutputFlag', False)

    # Step 1: define decision variables
    var_lambda = model.addVar(vtype=grb.GRB.CONTINUOUS)
    var_s = {}
    var_w = {}
    slack_var = {}
    for i in range(row):
        var_s[i] = model.addVar(vtype=grb.GRB.CONTINUOUS,)
    for j in range(col):
        var_w[j] = model.addVar(
            vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY)
        if pnorm == 1:
            slack_var[j] = model.addVar(vtype=grb.GRB.CONTINUOUS)

    # Step 2: integerate variables
    model.update()

    # Step 3: define constraints
    for i in range(row):
        model.addConstr(
            1 - y_train[i] * grb.quicksum(var_w[j] * x_train[i, j]
                                          for j in range(col)) <= var_s[i])
    if pnorm == 1:
        for j in range(col):
            model.addConstr(var_w[j] <= slack_var[j])
            model.addConstr(-var_w[j] <= slack_var[j])
        model.addConstr(grb.quicksum(slack_var[j]
                                     for j in range(col)) <= var_lambda)
    elif pnorm == 2:
        model.addQConstr(
            grb.quicksum(var_w[j] * var_w[j]
                         for j in range(col)) <= var_lambda * var_lambda)
    elif pnorm == float('Inf'):
        for j in range(col):
            model.addConstr(var_w[j] <= var_lambda)
            model.addConstr(-var_w[j] <= var_lambda)

    # Step 4: define objective value
    sum_var_s = grb.quicksum(var_s[i] for i in range(row))
    for epsilon in all_epsilon:
        obj = var_lambda * epsilon + 1 / row * sum_var_s
        model.setObjective(obj, grb.GRB.MINIMIZE)

        # Step 5: solve the problem
        model.optimize()

        # Step 6: store results
        w_opt = np.array([var_w[i].x for i in range(col)])
        tmp = {
            (float('Inf'), epsilon): {
                'w': w_opt,
                'objective': model.ObjVal,
                'diagnosis': model.status
            }
        }
        optimal.update(tmp)

    return optimal

def hinge_loss_minimization(data):
    """ classical SVM by minimizing hinge loss function solely """
    x_train = data['x']
    y_train = data['y'].flatten()

    row, col = x_train.shape

    # Step 0: create model
    model = grb.Model('classical_hinge_loss_minimization')
    model.setParam('OutputFlag', False)

    # Step 1: define decision variables
    var_s = {}
    var_w = {}
    for i in range(row):
        var_s[i] = model.addVar(vtype=grb.GRB.CONTINUOUS)
    for j in range(col):
        var_w[j] = model.addVar(
            vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY)

    # Step 2: integerate variables
    model.update()

    # Step 3: define constraints
    for i in range(row):
        model.addConstr(
            1 - y_train[i] * grb.quicksum(var_w[j] * x_train[i, j]
                                          for j in range(col)) <= var_s[i])

    # Step 4: define objective value
    sum_var_s = grb.quicksum(var_s[i] for i in range(row))
    obj = 1 / row * sum_var_s
    model.setObjective(obj, grb.GRB.MINIMIZE)

    # Step 5: solve the problem
    model.optimize()

    # Step 6: store results
    w_opt = np.array([var_w[i].x for i in range(col)])
    optimal = {
        'w': w_opt,
        'objective': model.ObjVal,
        'diagnosis': model.status
    }

    return optimal

def ksvm(param, data):
    """ kernelized SVM """
    certif = np.linalg.eigvalsh(data['K'])[0]
    if certif < 0:
        data['K'] = data['K'] - 2 * certif * np.eye(data['K'].shape[0])
    optimal = {}
    if len(param['kappa']) > 1 or float('inf') not in param['kappa']:
        optimal.update(dist_rob_ksvm(param, data))
    if float('Inf') in param['kappa']:
        optimal.update(regularized_ksvm(param, data))

    return optimal

def dist_rob_ksvm(param, data):
    """ kernelized distributionally robust SVM """
    kernel_train = data['K']
    y_train = data['y'].flatten()

    all_epsilon = list(param['epsilon'])
    all_epsilon.sort(reverse=True)
    all_kappa = list(param['kappa'])
    all_kappa.sort(reverse=True)
    if float('Inf') in all_kappa:
        all_kappa.remove(float('Inf'))
    if 0 in all_epsilon:
        all_epsilon.remove(0)

    row = kernel_train.shape[0]
    optimal = {}

    # Step 0: create model
    model = grb.Model('Ker_DRSVM')
    model.setParam('OutputFlag', False)

    # Step 1: define decision variables
    var_lambda = model.addVar(vtype=grb.GRB.CONTINUOUS)
    var_s = {}
    var_alpha = {}
    for i in range(row):
        var_s[i] = model.addVar(vtype=grb.GRB.CONTINUOUS)
        var_alpha[i] = model.addVar(
            vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY)

    # Step 2: integerate variables
    model.update()

    # Step 3: define constraints
    chg_cons = {}
    for i in range(row):
        model.addConstr(
            1 - y_train[i] * grb.quicksum(var_alpha[k] * kernel_train[k, i]
                                          for k in range(row)) <= var_s[i])
        chg_cons[i] = model.addConstr(
            1 + y_train[i] * grb.quicksum(var_alpha[k] * kernel_train[k, i]
                                          for k in range(row)) -
            all_kappa[0] * var_lambda <= var_s[i])
    model.addQConstr(
        grb.quicksum(var_alpha[k1] * kernel_train[k1, k2] * var_alpha[k2]
                     for k1 in range(row)
                     for k2 in range(row)) <= var_lambda * var_lambda)

    # Step 4: define objective value
    sum_var_s = grb.quicksum(var_s[i] for i in range(row))
    for index, kappa in enumerate(all_kappa):
        # Change model for different kappa
        if index > 0:
            for i in range(row):
                model.chgCoeff(chg_cons[i], var_lambda, -kappa)
        for epsilon in all_epsilon:
            obj = var_lambda * epsilon + 1 / row * sum_var_s
            model.setObjective(obj, grb.GRB.MINIMIZE)

            # Step 5: solve the problem
            model.optimize()

            # Step 6: store results
            alpha_opt = np.array([var_alpha[i].x for i in range(row)])
            tmp = {
                (kappa, epsilon): {
                    'alpha': alpha_opt,
                    'objective': model.ObjVal,
                    'diagnosis': model.status
                }
            }
            optimal.update(tmp)

    return optimal

def regularized_ksvm(param, data):
    """ kernelized robust/regularized SVM """
    kernel_train = data['K']
    y_train = data['y'].flatten()

    all_epsilon = list(param['epsilon'])
    all_epsilon.sort(reverse=True)
    if 0 in all_epsilon:
        all_epsilon.remove(0)

    row = kernel_train.shape[0]
    optimal = {}

    # Step 0: create model
    model = grb.Model('Ker_DRSVM')
    model.setParam('OutputFlag', False)

    # Step 1: define decision variables
    var_lambda = model.addVar(vtype=grb.GRB.CONTINUOUS)
    var_s = {}
    var_alpha = {}
    for i in range(row):
        var_s[i] = model.addVar(vtype=grb.GRB.CONTINUOUS)
        var_alpha[i] = model.addVar(
            vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY)

    # Step 2: integerate variables
    model.update()

    # Step 3: define constraints
    for i in range(row):
        model.addConstr(
            1 - y_train[i] * grb.quicksum(var_alpha[k] * kernel_train[k, i]
                                          for k in range(row)) <= var_s[i])
    model.addQConstr(
        grb.quicksum(var_alpha[k1] * kernel_train[k1, k2] * var_alpha[k2]
                     for k1 in range(row)
                     for k2 in range(row)) <= var_lambda * var_lambda)

    # Step 4: define objective value
    sum_var_s = grb.quicksum(var_s[i] for i in range(row))
    for epsilon in all_epsilon:
        obj = var_lambda * epsilon + 1 / row * sum_var_s
        model.setObjective(obj, grb.GRB.MINIMIZE)

        # Step 5: solve the problem
        model.optimize()

        # Step 6: store results
        alpha_opt = np.array([var_alpha[i].x for i in range(row)])
        tmp = {
            (float('Inf'), epsilon): {
                'alpha': alpha_opt,
                'objective': model.ObjVal,
                'diagnosis': model.status
            }
        }
        optimal.update(tmp)

    return optimal

def upper_bound_for_risk_estimation(param, data):
    """ Upper bound for risk estimation problem """
    x_train = data['x']
    y_train = data['y'].flatten()
    w_hat = data['w']

    pnorm = param['pnorm']
    all_epsilon = list(param['epsilon'])
    all_epsilon.sort(reverse=True)
    all_kappa = list(param['kappa'])
    all_kappa.sort(reverse=True)
    if float('Inf') in all_kappa:
        all_kappa.remove(float('Inf'))

    row = x_train.shape[0]
    optimal = {}

    # Step 0: create model
    model = grb.Model('upper_bound_for_risk_estimation')
    model.setParam('OutputFlag', False)

    # Step 1: define decision variables
    var_lambda = model.addVar(vtype=grb.GRB.CONTINUOUS)
    var_s = {}
    var_t = {}
    var_r = {}
    for i in range(row):
        var_s[i] = model.addVar(vtype=grb.GRB.CONTINUOUS)
        var_t[i] = model.addVar(vtype=grb.GRB.CONTINUOUS)
        var_r[i] = model.addVar(vtype=grb.GRB.CONTINUOUS)

    # Step 2: integerate variables
    model.update()

    # Step 3: define constraints
    chg_cons = {}
    for i in range(row):
        model.addConstr(
            1 - var_r[i] * y_train[i] * float(x_train[i, :].dot(w_hat)) <= var_s[i])
        chg_cons[i] = model.addConstr(
            1 + var_t[i] * y_train[i] * float(x_train[i, :].dot(w_hat)) -
            all_kappa[0] * var_lambda <= var_s[i])
        model.addConstr(var_r[i] * np.linalg.norm(w_hat, pnorm) <= var_lambda)
        model.addConstr(var_t[i] * np.linalg.norm(w_hat, pnorm) <= var_lambda)

    # Step 4: define objective value
    sum_var_s = grb.quicksum(var_s[i] for i in range(row))
    for index, kappa in enumerate(all_kappa):
        # Change model for different kappa
        if index > 0:
            for i in range(row):
                model.chgCoeff(chg_cons[i], var_lambda, -kappa)
        for epsilon in all_epsilon:
            obj = var_lambda * epsilon + 1 / row * sum_var_s
            model.setObjective(obj, grb.GRB.MINIMIZE)

            # Step 5: solve the problem
            model.optimize()

            # Step 6: store results
            tmp = {
                (kappa, epsilon): {
                    'upper_bound': model.ObjVal,
                    'diagnosis': model.status
                }
            }
            optimal.update(tmp)

    return optimal

def lower_bound_for_risk_estimation(param, data):
    """ Lower bound for risk estimation problem """
    x_train = data['x']
    y_train = data['y'].flatten()
    w_hat = data['w']

    pnorm = param['pnorm']
    all_epsilon = list(param['epsilon'])
    all_epsilon.sort(reverse=True)
    all_kappa = list(param['kappa'])
    all_kappa.sort(reverse=True)
    if float('Inf') in all_kappa:
        all_kappa.remove(float('Inf'))

    row = x_train.shape[0]
    optimal = {}

    # Step 0: create model
    model = grb.Model('lower_bound_for_risk_estimation')
    model.setParam('OutputFlag', False)

    # Step 1: define decision variables
    var_lambda = model.addVar(vtype=grb.GRB.CONTINUOUS)
    var_s = {}
    var_t = {}
    var_r = {}
    for i in range(row):
        var_s[i] = model.addVar(vtype=grb.GRB.CONTINUOUS)
        var_t[i] = model.addVar(vtype=grb.GRB.CONTINUOUS)
        var_r[i] = model.addVar(vtype=grb.GRB.CONTINUOUS)

    # Step 2: integerate variables
    model.update()

    # Step 3: define constraints
    chg_cons = {}
    for i in range(row):
        model.addConstr(
            1 + var_r[i] * y_train[i] * float(x_train[i, :].dot(w_hat)) <= var_s[i])
        chg_cons[i] = model.addConstr(
            1 - var_t[i] * y_train[i] * float(x_train[i, :].dot(w_hat)) -
            all_kappa[0] * var_lambda <= var_s[i])
        model.addConstr(var_r[i] * np.linalg.norm(w_hat, pnorm) <= var_lambda)
        model.addConstr(var_t[i] * np.linalg.norm(w_hat, pnorm) <= var_lambda)

    # Step 4: define objective value
    sum_var_s = grb.quicksum(var_s[i] for i in range(row))
    for index, kappa in enumerate(all_kappa):
        # Change model for different kappa
        if index > 0:
            for i in range(row):
                model.chgCoeff(chg_cons[i], var_lambda, -kappa)
        for epsilon in all_epsilon:
            obj = var_lambda * epsilon + 1 / row * sum_var_s
            model.setObjective(obj, grb.GRB.MINIMIZE)

            # Step 5: solve the problem
            model.optimize()

            # Step 6: store results
            tmp = {
                (kappa, epsilon): {
                    'lower_bound': 1 - model.ObjVal,
                    'diagnosis': model.status
                }
            }
            optimal.update(tmp)

    return optimal

def upper_bound_risk(param, data, w_hat):
    """ Upper bound for risk estimation problem """
    x_train = data['x']
    y_train = data['y'].flatten()

    pnorm = param['pnorm']
    all_epsilon = list(param['epsilon'])

    row = x_train.shape[0]
    optimal = {}

    # Step 0: create model
    model = grb.Model('upper_bound_risk')
    model.setParam('OutputFlag', False)

    # Step 1: define decision variables
    var_lambda = model.addVar(vtype=grb.GRB.CONTINUOUS)
    var_s = {}
    var_r = {}
    for i in range(row):
        var_s[i] = model.addVar(vtype=grb.GRB.CONTINUOUS)
        var_r[i] = model.addVar(vtype=grb.GRB.CONTINUOUS)

    # Step 2: integerate variables
    model.update()

    # Step 3: define constraints
    chg_cons = {}
    for i in range(row):
        model.addConstr(
            1 - var_r[i] * y_train[i] * float(x_train[i, :].dot(w_hat)) <= var_s[i])
        model.addConstr(var_r[i] * np.linalg.norm(w_hat, pnorm) <= var_lambda)

    # Step 4: define objective value
    sum_var_s = grb.quicksum(var_s[i] for i in range(row))

    optimal = np.zeros(len(all_epsilon))
    for index, epsilon in enumerate(all_epsilon):
        obj = var_lambda * epsilon + 1 / row * sum_var_s
        model.setObjective(obj, grb.GRB.MINIMIZE)

        # Step 5: solve the problem
        model.optimize()

        # Step 6: store results
        optimal[index] = model.ObjVal

    return optimal

def lower_bound_risk(param, data, w_hat):
    """ Lower bound for risk estimation problem """
    x_train = data['x']
    y_train = data['y'].flatten()

    pnorm = param['pnorm']
    all_epsilon = list(param['epsilon'])

    row = x_train.shape[0]
    optimal = {}

    # Step 0: create model
    model = grb.Model('lower_bound_risk')
    model.setParam('OutputFlag', False)

    # Step 1: define decision variables
    var_lambda = model.addVar(vtype=grb.GRB.CONTINUOUS)
    var_s = {}
    var_r = {}
    for i in range(row):
        var_s[i] = model.addVar(vtype=grb.GRB.CONTINUOUS)
        var_r[i] = model.addVar(vtype=grb.GRB.CONTINUOUS)

    # Step 2: integerate variables
    model.update()

    # Step 3: define constraints
    chg_cons = {}
    for i in range(row):
        model.addConstr(
            1 + var_r[i] * y_train[i] * float(x_train[i, :].dot(w_hat)) <= var_s[i])
        model.addConstr(var_r[i] * np.linalg.norm(w_hat, pnorm) <= var_lambda)

    # Step 4: define objective value
    sum_var_s = grb.quicksum(var_s[i] for i in range(row))

    optimal = np.zeros(len(all_epsilon))
    for index, epsilon in enumerate(all_epsilon):
        obj = var_lambda * epsilon + 1 / row * sum_var_s
        model.setObjective(obj, grb.GRB.MINIMIZE)

        # Step 5: solve the problem
        model.optimize()

        # Step 6: store results
        optimal[index] = 1 - model.ObjVal

    return optimal

def worst_case_distribution(data, w, epsilon, kappa, mat_c, vec_d):
    """ worst case distribution for hinge loss """
    x_train = data['x']
    y_train = data['y'].flatten()

    row, col = x_train.shape

    # Step 0: create model
    model = grb.Model('worst_case_distribution')
    model.setParam('OutputFlag', False)

    # Step 1: define decision variables
    alpha_p = {}
    alpha_n = {}
    var_q_p = {}
    var_q_n = {}
    slack_var_p = {}
    slack_var_n = {}
    for i in range(row):
        alpha_p[i, 0] = model.addVar(vtype=grb.GRB.CONTINUOUS)
        alpha_p[i, 1] = model.addVar(vtype=grb.GRB.CONTINUOUS)
        alpha_n[i, 0] = model.addVar(vtype=grb.GRB.CONTINUOUS)
        alpha_n[i, 1] = model.addVar(vtype=grb.GRB.CONTINUOUS)
        slack_var_p[i, 0] = model.addVar(vtype=grb.GRB.CONTINUOUS)
        slack_var_p[i, 1] = model.addVar(vtype=grb.GRB.CONTINUOUS)
        slack_var_n[i, 0] = model.addVar(vtype=grb.GRB.CONTINUOUS)
        slack_var_n[i, 1] = model.addVar(vtype=grb.GRB.CONTINUOUS)
        for j in range(col):
            var_q_p[i, j, 0] = model.addVar(vtype=grb.GRB.CONTINUOUS,
                                            lb=-grb.GRB.INFINITY)
            var_q_p[i, j, 1] = model.addVar(vtype=grb.GRB.CONTINUOUS,
                                            lb=-grb.GRB.INFINITY)
            var_q_n[i, j, 0] = model.addVar(vtype=grb.GRB.CONTINUOUS,
                                            lb=-grb.GRB.INFINITY)
            var_q_n[i, j, 1] = model.addVar(vtype=grb.GRB.CONTINUOUS,
                                            lb=-grb.GRB.INFINITY)

    # Step 2: integerate variables
    model.update()

    # Step 3: define constraints
    for i in range(row):
        model.addConstr(alpha_p[i, 0] + alpha_p[i, 1] + alpha_n[i, 0] + alpha_n[i, 1] == 1)
        for j in range(col):
            model.addConstr(var_q_p[i, j, 0] <= slack_var_p[i, 0])
            model.addConstr(-var_q_p[i, j, 0] <= slack_var_p[i, 0])
            model.addConstr(var_q_p[i, j, 1] <= slack_var_p[i, 1])
            model.addConstr(-var_q_p[i, j, 1] <= slack_var_p[i, 1])
            model.addConstr(var_q_n[i, j, 0] <= slack_var_n[i, 0])
            model.addConstr(-var_q_n[i, j, 0] <= slack_var_n[i, 0])
            model.addConstr(var_q_n[i, j, 1] <= slack_var_n[i, 1])
            model.addConstr(-var_q_n[i, j, 1] <= slack_var_n[i, 1])
        for k in range(len(vec_d)):
            model.addConstr(grb.quicksum(
                mat_c[k, j] * (x_train[i, j] * alpha_p[i, 0] + var_q_p[i, j, 0])
                for j in range(col)) <= vec_d[k] * alpha_p[i, 0]
                )
            model.addConstr(grb.quicksum(
                mat_c[k, j] * (x_train[i, j] * alpha_p[i, 1] + var_q_p[i, j, 1])
                for j in range(col)) <= vec_d[k] * alpha_p[i, 1]
                )
            model.addConstr(grb.quicksum(
                mat_c[k, j] * (x_train[i, j] * alpha_n[i, 0] + var_q_n[i, j, 0])
                for j in range(col)) <= vec_d[k] * alpha_n[i, 0]
                )
            model.addConstr(grb.quicksum(
                mat_c[k, j] * (x_train[i, j] * alpha_n[i, 1] + var_q_n[i, j, 1])
                for j in range(col)) <= vec_d[k] * alpha_n[i, 1]
                )
    model.addConstr(
        grb.quicksum(
            slack_var_p[i, 0] + slack_var_n[i, 0] + kappa * alpha_n[i, 0]
            for i in range(row)) + grb.quicksum(
                slack_var_p[i, 1] + slack_var_n[i, 1] + kappa * alpha_n[i, 1]
                for i in range(row)) <= row * epsilon)

    # Step 4: define objective value
    obj = 1 - 1.0 / row * grb.quicksum(
        (alpha_p[i, 0] - alpha_n[i, 0]) * y_train[i] * w[j] * x_train[i, j] + \
        y_train[i] * w[j] * (var_q_p[i, j, 0] - var_q_n[i, j, 0])
        for i in range(row)
        for j in range(col))
    model.setObjective(obj, grb.GRB.MAXIMIZE)

    # Step 5: solve the problem
    model.optimize()

    # Step 6: store results
    alpha_pos = np.array(
        [[alpha_p[i, 0].x for i in range(row)],
         [alpha_p[i, 1].x for i in range(row)]]).T
    alpha_neg = np.array(
        [[alpha_n[i, 0].x for i in range(row)],
         [alpha_n[i, 1].x for i in range(row)]]).T
    q_pos = np.zeros((x_train.shape[0], x_train.shape[1], 2))
    q_neg = np.zeros((x_train.shape[0], x_train.shape[1], 2))
    for i in range(row):
        for j in range(col):
            q_pos[i, j, 0] = var_q_p[i, j, 0].x
            q_pos[i, j, 1] = var_q_p[i, j, 1].x
            q_neg[i, j, 0] = var_q_n[i, j, 0].x
            q_neg[i, j, 1] = var_q_n[i, j, 1].x

    return alpha_pos, q_pos, alpha_neg, q_neg

def worst_case_distribution_inf(data, w, epsilon, mat_c, vec_d):
    """ worst case distribution for hinge loss """
    x_train = data['x']
    y_train = data['y'].flatten()

    row, col = x_train.shape

    # Step 0: create model
    model = grb.Model('worst_case_distribution')
    model.setParam('OutputFlag', False)

    # Step 1: define decision variables
    var_q_1 = {}
    var_q_2 = {}
    alpha_1 = {}
    alpha_2 = {}
    slack_var_1 = {}
    slack_var_2 = {}
    for i in range(row):
        slack_var_1[i] = model.addVar(vtype=grb.GRB.CONTINUOUS)
        slack_var_2[i] = model.addVar(vtype=grb.GRB.CONTINUOUS)
        alpha_1[i] = model.addVar(vtype=grb.GRB.CONTINUOUS)
        alpha_2[i] = model.addVar(vtype=grb.GRB.CONTINUOUS)
        for j in range(col):
            var_q_1[i, j] = model.addVar(vtype=grb.GRB.CONTINUOUS,
                                         lb=-grb.GRB.INFINITY)
            var_q_2[i, j] = model.addVar(vtype=grb.GRB.CONTINUOUS,
                                         lb=-grb.GRB.INFINITY)

    # Step 2: integerate variables
    model.update()

    # Step 3: define constraints
    for i in range(row):
        model.addConstr(alpha_1[i] + alpha_2[i] == 1)
        for j in range(col):
            model.addConstr(var_q_1[i, j] <= slack_var_1[i])
            model.addConstr(-var_q_1[i, j] <= slack_var_1[i])
            model.addConstr(var_q_2[i, j] <= slack_var_2[i])
            model.addConstr(-var_q_2[i, j] <= slack_var_2[i])
        for k in range(len(vec_d)):
            model.addConstr(grb.quicksum(
                mat_c[k, j] * (x_train[i, j] * alpha_1[i] + var_q_1[i, j])
                for j in range(col)) <= vec_d[k] * alpha_1[i]
                )
            model.addConstr(grb.quicksum(
                mat_c[k, j] * (x_train[i, j] * alpha_2[i] + var_q_2[i, j])
                for j in range(col)) <= vec_d[k] * alpha_2[i]
                )
    model.addConstr(grb.quicksum(
        slack_var_1[i] + slack_var_2[i] for i in range(row)) <= row * epsilon)

    # Step 4: define objective value
    obj = 1 - 1.0 / row * grb.quicksum(
        alpha_1[i] * y_train[i] * w[j] * x_train[i, j] + \
        y_train[i] * w[j] * var_q_1[i, j]
        for i in range(row)
        for j in range(col))
    model.setObjective(obj, grb.GRB.MAXIMIZE)

    # Step 5: solve the problem
    model.optimize()

    # Step 6: store results
    alpha_1 = np.array([alpha_1[i].x for i in range(row)])
    alpha_2 = np.array([alpha_2[i].x for i in range(row)])
    q_1 = np.zeros(x_train.shape)
    q_2 = np.zeros(x_train.shape)
    for i in range(row):
        for j in range(col):
            q_1[i, j] = var_q_1[i, j].x
            q_2[i, j] = var_q_2[i, j].x

    return alpha_1, q_1, alpha_2, q_2

