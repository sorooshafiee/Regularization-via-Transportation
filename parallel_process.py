""" Helper function for parallel computing """


from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Imputer
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel, laplacian_kernel
from sklearn.utils import shuffle
import pandas as pd
import dro_model


def parallel_classification_table1(x_train, y_train, x_test, y_test, param,
                                   kernel_functions, is_missing=False):
    """ This a function for using joblib to enhance parallel processing
    for the classification example in table2 """

    minmax_scaler = MinMaxScaler()
    x_train, x_test_sp, y_train, y_test_sp = train_test_split(
        x_train, y_train, train_size=500)
    x_test = np.vstack([x_test, x_test_sp]) if x_test.size else x_test_sp
    y_test = np.hstack([y_test, y_test_sp]) if y_test.size else y_test_sp

    if is_missing:
        for i in range(250):
            pix = np.random.permutation(784)
            pix = pix[0:588]
            x_train[i, pix] = np.nan
        impute = Imputer()
        x_train = impute.fit_transform(x_train)
    x_train = minmax_scaler.fit_transform(x_train)
    x_test = minmax_scaler.transform(x_test)

    # Initialize output
    dro_results = {}
    reg_results = {}

    for kernel_fun in kernel_functions:
        if kernel_fun.lower() == 'polynomial':
            best_params = []
            dro_score = []
            reg_score = []
            gamma = 1 / 100
            for deg in param['deg']:
                kernel_train = polynomial_kernel(x_train, degree=deg, gamma=gamma)
                total_score = validation_process(kernel_train, y_train, param)
                # Select the best model
                tot_score = pd.DataFrame(total_score)
                ave_score = tot_score.mean()
                best_kappa, best_epsilon = ave_score.idxmax()
                best_reg = ave_score[float('inf')].idxmax()
                tmp = {
                    'kappa': best_kappa,
                    'epsilon': best_epsilon,
                    'reg': best_reg
                }
                best_params.append(tmp)
                dro_score.append(ave_score[(best_kappa, best_epsilon)])
                reg_score.append(ave_score[(float('inf'), best_reg)])
            sel_idx = dro_score.index(max(dro_score))
            sel_deg = param['deg'][sel_idx]
            sel_kernel_train = polynomial_kernel(x_train, degree=sel_deg, gamma=gamma)
            sel_kernel_test = polynomial_kernel(x_test, x_train, degree=sel_deg, gamma=gamma)
            sel_param = {
                'epsilon': [best_params[sel_idx]['epsilon']],
                'kappa': [best_params[sel_idx]['kappa']]
            }
            dro_results[kernel_fun] = test_performance(
                sel_kernel_train, y_train, sel_kernel_test, y_test, sel_param)

            sel_idx = reg_score.index(max(reg_score))
            sel_deg = param['deg'][sel_idx]
            sel_kernel_train = polynomial_kernel(x_train, degree=sel_deg, gamma=gamma)
            sel_kernel_test = polynomial_kernel(x_test, x_train, degree=sel_deg, gamma=gamma)
            sel_param = {
                'epsilon': [best_params[sel_idx]['reg']],
                'kappa': [float('inf')]
            }
            reg_results[kernel_fun] = test_performance(
                sel_kernel_train, y_train, sel_kernel_test, y_test, sel_param)

        elif (kernel_fun.lower() == 'rbf') or (kernel_fun.lower() == 'gaussian'):
            best_params = []
            dro_score = []
            reg_score = []
            for gamma in param['gamma_rbf']:
                kernel_train = rbf_kernel(x_train, gamma=gamma)
                total_score = validation_process(kernel_train, y_train, param)
                # Select the best model
                tot_score = pd.DataFrame(total_score)
                ave_score = tot_score.mean()
                best_kappa, best_epsilon = ave_score.idxmax()
                best_reg = ave_score[float('inf')].idxmax()
                tmp = {
                    'kappa': best_kappa,
                    'epsilon': best_epsilon,
                    'reg': best_reg
                }
                best_params.append(tmp)
                dro_score.append(ave_score[(best_kappa, best_epsilon)])
                reg_score.append(ave_score[(float('inf'), best_reg)])
            sel_idx = dro_score.index(max(dro_score))
            sel_gamma = param['gamma_rbf'][sel_idx]
            sel_kernel_train = rbf_kernel(x_train, gamma=sel_gamma)
            sel_kernel_test = rbf_kernel(x_test, x_train, gamma=sel_gamma)
            sel_param = {
                'epsilon': [best_params[sel_idx]['epsilon']],
                'kappa': [best_params[sel_idx]['kappa']]
            }
            dro_results[kernel_fun] = test_performance(
                sel_kernel_train, y_train, sel_kernel_test, y_test, sel_param)

            sel_idx = reg_score.index(max(reg_score))
            sel_gamma = param['gamma_rbf'][sel_idx]
            sel_kernel_train = rbf_kernel(x_train, gamma=sel_gamma)
            sel_kernel_test = rbf_kernel(x_test, x_train, gamma=sel_gamma)
            sel_param = {
                'epsilon': [best_params[sel_idx]['reg']],
                'kappa': [float('inf')]
            }
            reg_results[kernel_fun] = test_performance(
                sel_kernel_train, y_train, sel_kernel_test, y_test, sel_param)

        elif kernel_fun.lower() == 'laplacian':
            best_params = []
            dro_score = []
            reg_score = []
            for gamma in param['gamma_lap']:
                kernel_train = laplacian_kernel(x_train, gamma=gamma)
                total_score = validation_process(kernel_train, y_train, param)
                # Select the best model
                tot_score = pd.DataFrame(total_score)
                ave_score = tot_score.mean()
                best_kappa, best_epsilon = ave_score.idxmax()
                best_reg = ave_score[float('inf')].idxmax()
                tmp = {
                    'kappa': best_kappa,
                    'epsilon': best_epsilon,
                    'reg': best_reg
                }
                best_params.append(tmp)
                dro_score.append(ave_score[(best_kappa, best_epsilon)])
                reg_score.append(ave_score[(float('inf'), best_reg)])
            sel_idx = dro_score.index(max(dro_score))
            sel_gamma = param['gamma_lap'][sel_idx]
            sel_kernel_train = laplacian_kernel(x_train, gamma=sel_gamma)
            sel_kernel_test = laplacian_kernel(x_test, x_train, gamma=sel_gamma)
            sel_param = {
                'epsilon': [best_params[sel_idx]['epsilon']],
                'kappa': [best_params[sel_idx]['kappa']]
            }
            dro_results[kernel_fun] = test_performance(
                sel_kernel_train, y_train, sel_kernel_test, y_test, sel_param)

            sel_idx = reg_score.index(max(reg_score))
            sel_gamma = param['gamma_lap'][sel_idx]
            sel_kernel_train = laplacian_kernel(x_train, gamma=sel_gamma)
            sel_kernel_test = laplacian_kernel(x_test, x_train, gamma=sel_gamma)
            sel_param = {
                'epsilon': [best_params[sel_idx]['reg']],
                'kappa': [float('inf')]
            }
            reg_results[kernel_fun] = test_performance(
                sel_kernel_train, y_train, sel_kernel_test, y_test, sel_param)
        else:
            raise 'Undefined kernel function'
    return (dro_results, reg_results)


def parallel_classification_table2(*args):
    """ This a function for using joblib to enhance parallel processing
    for the classification example in table1 """

    # Setting parameters
    all_param = {
        'epsilon': [1e-4, 5e-4, 1e-3, 5e-2, 1e-2, 5e-2, 1e-1],
        'kappa': [0.1, 0.2, 0.3, 0.4, 0.5, 1, float('inf')],
        'd': [],
        'C': []
    }
    pnorms = [1, 2, float('Inf')]

    # Initialize output
    DRSVM_AUC = {}
    RSVM_AUC = {}

    # Load input data
    nargin = len(args)
    if nargin == 2:
        x_data = args[0]
        y_data = args[1]
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.25)
    elif nargin == 4:
        x_train = args[0]
        y_train = args[1]
        x_train, y_train = shuffle(x_train, y_train)
        x_test = args[2]
        y_test = args[3]

    # Fit classical svm model, hinge loss minimization
    stand_scaler = StandardScaler()
    x_train_nrm = stand_scaler.fit_transform(x_train)
    x_test_nrm = stand_scaler.transform(x_test)
    training_data = {'x': x_train_nrm, 'y': y_train}
    optimal = dro_model.hinge_loss_minimization(training_data)
    w_opt = optimal['w']
    y_scores = 1 / (1 + np.exp(-x_test_nrm.dot(w_opt)))
    SVM_AUC = roc_auc_score(y_test, y_scores)

    # Parameter selection and then test the model performance
    skf = StratifiedKFold(n_splits=5)
    for pnorm in pnorms:
        all_param['pnorm'] = pnorm
        total_score = defaultdict(list)
        # K-fold cross validation
        for train_index, val_index in skf.split(x_train, y_train):
            x_train_k, x_val_k = x_train[train_index], x_train[val_index]
            y_train_k, y_val_k = y_train[train_index], y_train[val_index]
            x_train_k = stand_scaler.fit_transform(x_train_k)
            x_val_k = stand_scaler.transform(x_val_k)
            data_k = {'x': x_train_k, 'y': y_train_k}
            optimal = dro_model.svm(all_param, data_k)
            for key, value in optimal.items():
                w_opt = np.array(value['w'])
                y_scores = 1 / (1 + np.exp(-x_val_k.dot(w_opt)))
                total_score[key].append(roc_auc_score(y_val_k, y_scores))
        # Select the best model
        tot_score = pd.DataFrame(total_score)
        ave_score = tot_score.mean()
        best_kappa, best_epsilon = ave_score.idxmax()
        best_reg = ave_score[float('inf')].idxmax()

        param = {
            'epsilon': [best_epsilon],
            'kappa': [best_kappa],
            'pnorm': pnorm,
            'C': [],
            'd': []
        }
        optimal = dro_model.svm(param, training_data)
        w_opt = optimal[(best_kappa, best_epsilon)]['w']
        y_scores = 1 / (1 + np.exp(-x_test_nrm.dot(w_opt)))
        DRSVM_AUC[pnorm] = roc_auc_score(y_test, y_scores)

        param = {
            'epsilon': [best_reg],
            'kappa': [float('inf')],
            'pnorm': pnorm,
            'C': [],
            'd': []
        }
        optimal = dro_model.svm(param, training_data)
        w_opt = optimal[(float('inf'), best_reg)]['w']
        y_scores = 1 / (1 + np.exp(-x_test_nrm.dot(w_opt)))
        RSVM_AUC[pnorm] = roc_auc_score(y_test, y_scores)

    return (DRSVM_AUC, RSVM_AUC, SVM_AUC)

def validation_process(kernel_train, y_train, param):
    """ This function applies 5-fold cross validation and return
    the score on the validation sets """
    total_score = defaultdict(list)
    skf = StratifiedKFold(n_splits=5)
    # K-fold cross validation
    for train_index, val_index in skf.split(kernel_train, y_train):
        ker_train_k = kernel_train[train_index, :][:, train_index]
        ker_val_k = kernel_train[val_index, :][:, train_index]
        y_train_k, y_val_k = y_train[train_index], y_train[val_index]
        data_k = {'K': ker_train_k, 'y': y_train_k}
        optimal = dro_model.ksvm(param, data_k)
        for key, value in optimal.items():
            alpha_opt = np.array(value['alpha'])
            y_pred = np.sign(ker_val_k.dot(alpha_opt))
            total_score[key].append(accuracy_score(y_val_k, y_pred))
    return total_score

def test_performance(kernel_train, y_train, kernel_test, y_test, param):
    """ Re-train the model with all data and return the performance
    on the test dataset """
    training_data = {'K': kernel_train, 'y': y_train}
    optimal = dro_model.ksvm(param, training_data)
    alpha_opt = optimal[(param['kappa'][0], param['epsilon'][0])]['alpha']
    y_pred = np.sign(kernel_test.dot(alpha_opt))
    return accuracy_score(y_test, y_pred)

def parallel_classification_figure(x_train, y_train, x_test, y_test, param):
    """ This a function for using joblib to enhance parallel processing
    for classification example in figure1 """
    minmax_scaler = MinMaxScaler()
    x_train_i, x_val_i, y_train_i, y_val_i = train_test_split(
        x_train, y_train, train_size=500)
    x_test_i = np.vstack([x_test, x_val_i]) if x_test.size else x_val_i
    y_test_i = np.hstack([y_test, y_val_i]) if y_test.size else y_val_i

    x_train_i = minmax_scaler.fit_transform(x_train_i)
    x_test_i = minmax_scaler.transform(x_test_i)

    training_data = {'x': x_train_i, 'y': y_train_i}
    optimal = dro_model.svm(param, training_data)
    total_accuracy = {}
    total_roc = {}
    total_loss = {}
    for key, value in optimal.items():
        w_opt = np.array(value['w'])
        y_pred = np.sign(x_test_i.dot(w_opt))
        y_score = 1 / (1 + np.exp(-x_test_i.dot(w_opt)))
        total_accuracy[key] = accuracy_score(y_test_i, y_pred)
        total_roc[key] = roc_auc_score(y_test_i, y_score)
        total_loss[key] = np.mean(np.maximum(1 - y_test_i * x_test_i.dot(w_opt), 0))
    return (total_accuracy, total_roc, total_loss)
