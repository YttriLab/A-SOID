import math
import numpy as np

from asoid.utils.extract_features_2D import bsoid_extract_numba
from asoid.utils.extract_features_3D import asoid_extract_numba_3d

def bsoid_predict_numba(feats, scaler, clf):
    """
    :param scaler:
    :param feats: list, multiple feats (original feature space)
    :param clf: Obj, MLP classifier
    :return nonfs_labels: list, label/100ms
    LEGACY FUNCTION
    """
    labels_fslow = []
    for i in range(0, len(feats)):
        scaled_feats = scaler.transform(feats[i])
        labels = clf.predict(np.nan_to_num(scaled_feats))
        labels_fslow.append(labels)
    return labels_fslow


def asoid_predict_numba(feats, scaler, clf):
    """
    :param feats: list, multiple feats (original feature space)
    :param clf: Obj, MLP classifier
    :return nonfs_labels: list, label/100ms
    """
    labels_fslow = []
    for i in range(0, len(feats)):
        scaled_feats = scaler.transform(feats[i])
        labels = clf.predict(np.nan_to_num(scaled_feats))
        labels_fslow.append(labels)
    return labels_fslow

def bsoid_predict_numba_noscale(scaled_feats, clf):
    """
    :param scaler:
    :param feats: list, multiple feats (original feature space)
    :param clf: Obj, MLP classifier
    :return nonfs_labels: list, label/100ms
    """
    labels_fslow = []
    for i in range(0, len(scaled_feats)):
        labels = clf.predict(np.nan_to_num(scaled_feats[i]))
        labels_fslow.append(labels)
    return labels_fslow


def bsoid_predict_proba_numba_noscale(scaled_feats, clf):
    """
    :param scaler:
    :param feats: list, multiple feats (original feature space)
    :param clf: Obj, MLP classifier
    :return nonfs_labels: list, label/100ms
    """
    labels_proba = []
    for i in range(0, len(scaled_feats)):
        predict_proba = clf.predict_proba(np.nan_to_num(scaled_feats[i]))
        labels_proba.append(predict_proba)
    return labels_proba


def frameshift_predict(data_test, num_test, scaler, rf_model, framerate=30):
    labels_fs = []
    new_predictions = []
    for i in range(num_test):
        feats_new = bsoid_extract_numba([data_test[i]], framerate)
        labels = asoid_predict_numba(feats_new, scaler, rf_model)
        for m in range(0, len(labels)):
            labels[m] = labels[m][::-1]
        labels_pad = -1 * np.ones([len(labels), len(max(labels, key=lambda x: len(x)))])
        for n, l in enumerate(labels):
            labels_pad[n][0:len(l)] = l
            labels_pad[n] = labels_pad[n][::-1]
            if n > 0:
                labels_pad[n][0:n] = labels_pad[n - 1][0:n]
        labels_fs.append(labels_pad.astype(int))
    for k in range(0, len(labels_fs)):
        labels_fs2 = []
        for l in range(math.floor(framerate / 10)):
            labels_fs2.append(labels_fs[k][l])
        new_predictions.append(np.array(labels_fs2).flatten('F'))
    new_predictions_pad = []
    for i in range(0, len(new_predictions)):
        new_predictions_pad.append(np.pad(new_predictions[i], (len(data_test[i]) -
                                                               len(new_predictions[i]), 0), 'edge'))
    return np.hstack(new_predictions_pad)


def bsoid_predict_proba_numba(feats, scaler, clf):
    """
    :param scaler:
    :param feats: list, multiple feats (original feature space)
    :param clf: Obj, MLP classifier
    :return nonfs_labels: list, label/100ms
    """
    labels_fslow = []
    proba_fslow = []
    for i in range(0, len(feats)):
        scaled_feats = scaler.transform(feats[i])
        labels = clf.predict(np.nan_to_num(scaled_feats))
        proba = clf.predict_proba(np.nan_to_num(scaled_feats))
        labels_fslow.append(labels)
        proba_fslow.append(proba)
    return labels_fslow, proba_fslow


def frameshift_predict_proba(data_test, num_test, scaler, rf_model, framerate=120):
    labels_fs = []
    proba_fs = []
    new_predictions = []
    new_proba = []
    for i in range(num_test):
        feats_new = bsoid_extract_numba([data_test[i]], framerate)
        labels, proba = bsoid_predict_proba_numba(feats_new, scaler, rf_model)
        for m in range(0, len(labels)):
            labels[m] = labels[m][::-1]
            proba[m] = proba[m][::-1, :]
        labels_pad = -1 * np.ones([len(labels), len(max(labels, key=lambda x: len(x)))])
        proba_pad = -1 * np.ones([len(proba), len(max(proba, key=lambda x: len(x))), proba[0].shape[1]])
        for n, l in enumerate(labels):
            labels_pad[n][0:len(l)] = l
            labels_pad[n] = labels_pad[n][::-1]
            if n > 0:
                labels_pad[n][0:n] = labels_pad[n - 1][0:n]
        for n2, l2 in enumerate(proba):
            proba_pad[n2][0:len(l2), :] = l2
            proba_pad[n2] = proba_pad[n2][::-1, :]
            if n2 > 0:
                proba_pad[n2][0:n2, :] = proba_pad[n2 - 1][0:n2, :]
        labels_fs.append(labels_pad.astype(int))
        proba_fs.append(proba_pad.astype(float))
    for k in range(0, len(labels_fs)):
        labels_fs2 = []
        proba_fs2 = []
        for l in range(math.floor(framerate / 10)):
            labels_fs2.append(labels_fs[k][l])
            proba_fs2.append(proba_fs[k][l])
        new_predictions.append(np.array(labels_fs2).flatten('F'))
        new_proba.append(np.array(proba_fs2).reshape(-1, np.array(proba_fs2).shape[2]))
    new_predictions_pad = []
    new_proba_pad = []
    for i in range(0, len(new_predictions)):
        new_predictions_pad.append(np.pad(new_predictions[i], (len(data_test[i]) -
                                                               len(new_predictions[i]), 0), 'edge'))
        new_proba_pad.append(np.pad(new_proba[i], [(len(data_test[i]) -
                                                    len(new_proba[i]), 0), (0, 0)], 'edge'))
    return np.hstack(new_predictions_pad), np.vstack(new_proba_pad)



def frameshift_predict_3d(data_test, num_test, scaler, rf_model, framerate):
    labels_fs = []
    new_predictions = []
    for i in range(num_test):
        feats_new = asoid_extract_numba_3d([data_test[i]], framerate)
        labels = asoid_predict_numba(feats_new, scaler, rf_model)
        for m in range(0, len(labels)):
            labels[m] = labels[m][::-1]
        labels_pad = -1 * np.ones([len(labels), len(max(labels, key=lambda x: len(x)))])
        for n, l in enumerate(labels):
            labels_pad[n][0:len(l)] = l
            labels_pad[n] = labels_pad[n][::-1]
            if n > 0:
                labels_pad[n][0:n] = labels_pad[n - 1][0:n]
        labels_fs.append(labels_pad.astype(int))
    for k in range(0, len(labels_fs)):
        labels_fs2 = []
        for l in range(math.floor(framerate / 10)):
            labels_fs2.append(labels_fs[k][l])
        new_predictions.append(np.array(labels_fs2).flatten('F'))
    new_predictions_pad = []
    for i in range(0, len(new_predictions)):
        new_predictions_pad.append(np.pad(new_predictions[i], (len(data_test[i]) -
                                                               len(new_predictions[i]), 0), 'edge'))
    return np.hstack(new_predictions_pad)


def weighted_smoothing(predictions, size):
    predictions_new = predictions.copy()
    group_start = [0]
    group_start = np.hstack((group_start, np.where(np.diff(predictions) != 0)[0] + 1))
    for i in range(len(group_start) - 3):
        # sandwich jitters within a bout (jitter size defined by size)
        if group_start[i + 2] - group_start[i + 1] < size:
            if predictions_new[group_start[i + 2]] == predictions_new[group_start[i]] and \
                    predictions_new[group_start[i]:group_start[i + 1]].shape[0] >= size and \
                    predictions_new[group_start[i + 2]:group_start[i + 3]].shape[0] >= size:
                predictions_new[group_start[i]:group_start[i + 2]] = predictions_new[group_start[i]]

    for i in range(len(group_start) - 3):
        # replace jitter by previous behavior when it does not reach size
        if group_start[i + 1] - group_start[i] < size:
            predictions_new[group_start[i]:group_start[i + 1]] = predictions_new[group_start[i] - 1]
    return predictions_new