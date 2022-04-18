import math

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from math import log


def tpr(tn, fp, fn, tp):
    if tp+fn != 0:
        return float(tp / (tp+fn))
    else:
        return 0


def tnr(tn, fp, fn, tp):
    if tn+fp != 0:
        return float(tn / (tn+fp))
    else:
        return 0


def min_max_norm(minimum, maximum, values):
    return (min(values) - minimum) / (maximum - minimum)


def var_norm(data):
    np_data = np.asarray(data)
    variance = np_data.var(axis=0)
    return float(1-(variance / 0.5))


def er(tn, fp, fn, tp):
    return float((fp+fn) / 2)


def lr(tn, fp, fn, tp):
    temp = 1-tpr(tn, fp, fn, tp)
    if temp != 0:
        return float(tpr(tn, fp, fn, tp) / (1-tpr(tn, fp, fn, tp)))
    else:
        return 0


def precision(tn, fp, fn, tp):
    if tp+fp != 0:
        return float(tp / (tp+fp))
    else:
        return 0


def get_value_records(value, data, y_pred):
    value_records = pd.DataFrame(data.loc[data[data.columns[0]] == value])
    value_pred = pd.DataFrame(y_pred.loc[value_records.index])
    return value_records, value_pred


def get_not_value_records(value, data, y_pred):
    value_records = pd.DataFrame(data.loc[data[data.columns[0]] != value])
    value_pred = pd.DataFrame(y_pred.loc[value_records.index])
    return value_records, value_pred


def treat(tn, fp, fn, tp):
    if fp != 0:
        return float(fn / fp)
    else:
        return 0


def enp(tn, fp, fn, tp):
    if tn+fn != 0:
        return float(tn / (tn+fn))
    else:
        return 0


def acc(tn, fp, fn, tp):
    if tn+fn+tp+fp != 0:
        return float((tn+tp)/ (tn+fn+tp+fp))
    else:
        return 0


def cm_value_n_value(value, protected, y_pred, y_true, possible_labels):
    value_records, value_pred = get_value_records(value, protected, y_pred)
    value_true = pd.DataFrame(y_true.loc[value_records.index])
    vtn, vfp, vfn, vtp = confusion_matrix(value_true, value_pred, labels=possible_labels).ravel()
    not_value_records, not_value_pred = get_not_value_records(value, protected, y_pred)
    not_value_true = pd.DataFrame(y_true.loc[not_value_records.index])
    ntn, nfp, nfn, ntp = confusion_matrix(not_value_true, not_value_pred, labels=possible_labels).ravel()
    return vtn, vfp, vfn, vtp, ntn, nfp, nfn, ntp


def fpr(tn, fp, fn, tp):
    if tn+fp != 0:
        return float(fp / (tn+fp))
    else:
        return 0


def fnr(tn, fp, fn, tp):
    if tp+fn != 0:
        return float(fn / (tp+fn))
    else:
        return 0


def mutual_information_h(x_values: pd.DataFrame):
    h = 0.0
    for x in x_values[x_values.columns[0]].unique():
        x_records = x_values[x_values[x_values.columns[0]] == x]
        if x_values.size != 0:
            px = x_records.size / x_values.size
            logpx = log(px)
            h = h+(px*logpx)
    h = -1*h
    return h


def mutual_information_mi(y_pred, protected, positive):
    MI_list = []
    i = 0.0
    protected_values = protected[protected.columns[0]].unique()
    y_pred[y_pred.columns[0]] = [-1 if x != positive else x for x in y_pred[y_pred.columns[0]]]
    posible_labels = [positive, -1]
    for value in protected_values:
        temp_prot = protected.copy()
        temp_prot[temp_prot.columns[0]] = [-1 if x != value else x for x in temp_prot[temp_prot.columns[0]]]
        value_records, value_pred = get_value_records(value, temp_prot, y_pred)
        if temp_prot.size == 0:
            p_value = 0
        else:
            p_value = value_records.size / temp_prot.size
        for label in posible_labels:
            p_label = y_pred[y_pred[y_pred.columns[0]] == label].size / y_pred.size
            p_label_value = value_pred[value_pred[value_pred.columns[0]] == label].size / y_pred.size
            if p_value == 0 or p_label ==0:
                inlog =0
            else:
                inlog = p_label_value/(p_value*p_label)
            if inlog != 0:
                i = i + abs(p_label_value*log(inlog))
        h_pred = mutual_information_h(y_pred)
        h_prot = mutual_information_h(temp_prot)
        if h_pred*h_prot != 0:
            MI_list.append(i / math.sqrt(h_pred*h_prot))
        else:
            MI_list.append(0)
        i = 0.0
    return MI_list


