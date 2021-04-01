import numpy as np


def accuracy_score(y_true, y_predict, percent=None):
    # Подготка выборки согласно условиям
    if percent != None:
        y_true = y_true[:y_true.shape[0] * percent // 100]
        y_predict = y_predict[:y_predict.shape[0] * percent // 100]
    else:
        percent = 50
    # Разделение на TP, FP,FN, TN
    y_binar = ((y_predict[:,0] >= percent / 100)) * 1
    count_one_zer_true = np.sum(y_true == 1)
    count_one_zer_pred = np.sum((y_predict[:, 0] == 1) & (y_true == y_predict[:, 0]))
    if count_one_zer_pred / count_one_zer_true < 0.5:
        y_binar = ((y_predict[:,1] >= percent / 100)) * 1
    TP = np.sum((y_binar == 1) & (y_binar == y_true))
    FP = np.sum(y_binar == 1) - TP
    TN = np.sum((y_binar == 0) & (y_binar == y_true))
    FN = np.sum(y_binar == 0) - TN
    return (TP + TN) / (TP + TN + FP + FN)


def precision_score(y_true, y_predict, percent=None):
    # Подготка выборки согласно условиям
    if percent != None:
        y_true = y_true[:y_true.shape[0] * percent // 100]
        y_predict = y_predict[:y_predict.shape[0] * percent // 100]
    else:
        percent = 50
    # Разделение на TP, FP,FN, TN
    y_binar = ((y_predict[:,0] >= percent / 100)) * 1
    count_one_zer_true = np.sum(y_true == 1)
    count_one_zer_pred = np.sum((y_predict[:, 0] == 1) & (y_true == y_predict[:, 0]))
    if count_one_zer_pred / count_one_zer_true < 0.5:
        y_binar = ((y_predict[:,1] >= percent / 100)) * 1
    TP = np.sum((y_binar == 1) & (y_binar == y_true))
    FP = np.sum(y_binar == 1) - TP
    TN = np.sum((y_binar == 0) & (y_binar == y_true))
    FN = np.sum(y_binar == 0) - TN
    return TP / (TP + FP)


def recall_score(y_true, y_predict, percent=None):
    # Подготка выборки согласно условиям
    if percent != None:
        y_true = y_true[:y_true.shape[0] * percent // 100]
        y_predict = y_predict[:y_predict.shape[0] * percent // 100]
    else:
        percent = 50
    # Разделение на TP, FP,FN, TN
    y_binar = ((y_predict[:,0] >= percent / 100)) * 1
    count_one_zer_true = np.sum(y_true == 1)
    count_one_zer_pred = np.sum((y_predict[:, 0] == 1) & (y_true == y_predict[:, 0]))
    if count_one_zer_pred / count_one_zer_true < 0.5:
        y_binar = ((y_predict[:,1] >= percent / 100)) * 1
    TP = np.sum((y_binar == 1) & (y_binar == y_true))
    FP = np.sum(y_binar == 1) - TP
    TN = np.sum((y_binar == 0) & (y_binar == y_true))
    FN = np.sum(y_binar == 0) - TN
    return TP / (TP + FN)


def lift_score(y_true, y_predict, percent=None):
    # Подготка выборки согласно условиям
    if percent != None:
        y_true = y_true[:y_true.shape[0] * percent // 100]
        y_predict = y_predict[:y_predict.shape[0] * percent // 100]
    else:
        percent = 50
    # Разделение на TP, FP,FN, TN
    y_binar = ((y_predict[:,0] >= percent / 100)) * 1
    count_one_zer_true = np.sum(y_true == 1)
    count_one_zer_pred = np.sum((y_predict[:, 0] == 1) & (y_true == y_predict[:, 0]))
    if count_one_zer_pred / count_one_zer_true < 0.5:
        y_binar = ((y_predict[:,1] >= percent / 100)) * 1
    TP = np.sum((y_binar == 1) & (y_binar == y_true))
    FP = np.sum(y_binar == 1) - TP
    TN = np.sum((y_binar == 0) & (y_binar == y_true))
    FN = np.sum(y_binar == 0) - TN
    return TP / (TP + FP) / (TP + FN) / y_true.shape[0]


def f1_score(y_true, y_predict, percent=None):
    # Подготка выборки согласно условиям
    if percent != None:
        y_true = y_true[:y_true.shape[0] * percent // 100]
        y_predict = y_predict[:y_predict.shape[0] * percent // 100]
    else:
        percent = 50
    # Разделение на TP, FP,FN, TN
    y_binar = ((y_predict[:,0] >= percent / 100)) * 1
    count_one_zer_true = np.sum(y_true == 1)
    count_one_zer_pred = np.sum((y_predict[:, 0] == 1) & (y_true == y_predict[:, 0]))
    if count_one_zer_pred / count_one_zer_true < 0.5:
        y_binar = ((y_predict[:,1] >= percent / 100)) * 1
    TP = np.sum((y_binar == 1) & (y_binar == y_true))
    FP = np.sum(y_binar == 1) - TP
    TN = np.sum((y_binar == 0) & (y_binar == y_true))
    FN = np.sum(y_binar == 0) - TN
    return 2 * (TP / (TP + FP) * TP / (TP + FN)) / (TP / (TP + FP) + TP / (TP + FN))



file = np.loadtxt('lecture02\Homework\Problem2\HW2_labels.txt',  delimiter=',')
y_predict, y_true = file[:, :2], file[:, -1]
print(y_true.shape)

print('accuracy_score = ', accuracy_score(y_true, y_predict, percent=None))
print('precision_score = ', precision_score(y_true, y_predict, percent=None))
print('recall_score = ', recall_score(y_true, y_predict, percent=None))
print('lift_score = ', lift_score(y_true, y_predict, percent=None))
print('f1_score = ', f1_score(y_true, y_predict, percent=None))
