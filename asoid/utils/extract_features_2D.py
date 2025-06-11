import numpy as np
from numba import jit
from numba.typed import List
from sklearn.preprocessing import StandardScaler



@jit(nopython=True)
def fast_standardize(data):
    a_ = (data - np.mean(data)) / np.std(data)
    return a_


def fast_nchoose2(n, k):
    a = np.ones((k, n - k + 1), dtype=int)
    a[0] = np.arange(n - k + 1)
    for j in range(1, k):
        reps = (n - k + j) - a[j - 1]
        a = np.repeat(a, reps, axis=1)
        ind = np.add.accumulate(reps)
        a[j, ind[:-1]] = 1 - reps[1:]
        a[j, 0] = j
        a[j] = np.add.accumulate(a[j])
        return a


@jit(nopython=True)
def fast_running_mean(x, N):
    out = np.zeros_like(x, dtype=np.float64)
    dim_len = x.shape[0]
    for i in range(dim_len):
        if N % 2 == 0:
            a, b = i - (N - 1) // 2, i + (N - 1) // 2 + 2
        else:
            a, b = i - (N - 1) // 2, i + (N - 1) // 2 + 1
        a = max(0, a)
        b = min(dim_len, b)
        out[i] = np.mean(x[a:b])
    return out


@jit(nopython=True)
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


@jit(nopython=True)
def np_mean(array, axis):
    return np_apply_along_axis(np.mean, axis, array)


@jit(nopython=True)
def np_std(array, axis):
    return np_apply_along_axis(np.std, axis, array)


@jit(nopython=True)
def angle_between(vector1, vector2):
    """ Returns the angle in radians between given vectors"""
    v1_u = unit_vector(vector1)
    v2_u = unit_vector(vector2)
    minor = np.linalg.det(
        np.stack((v1_u[-2:], v2_u[-2:]))
    )
    if minor == 0:
        sign = 1
    else:
        sign = -np.sign(minor)
    dot_p = np.dot(v1_u, v2_u)
    dot_p = min(max(dot_p, -1.0), 1.0)
    return sign * np.arccos(dot_p)


@jit(nopython=True)
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


@jit(nopython=True)
def fast_displacment(data, reduce=False):
    data_length = data.shape[0]
    if reduce:
        displacement_array = np.zeros((data_length, int(data.shape[1] / 10)), dtype=np.float64)
    else:
        displacement_array = np.zeros((data_length, int(data.shape[1] / 2)), dtype=np.float64)
    for r in range(data_length):
        if r < data_length - 1:
            if reduce:
                count = 0
                for c in range(int(data.shape[1] / 2 - 2), data.shape[1], int(data.shape[1] / 2)):
                    displacement_array[r, count] = np.linalg.norm(data[r + 1, c:c + 2] - data[r, c:c + 2])
                    count += 1
            else:
                for c in range(0, data.shape[1], 2):
                    displacement_array[r, int(c / 2)] = np.linalg.norm(data[r + 1, c:c + 2] - data[r, c:c + 2])
    return displacement_array


@jit(nopython=True)
def fast_length_angle(data, index):
    data_length = data.shape[0]
    length_2d_array = np.zeros((data_length, index.shape[1], 2), dtype=np.float64)
    for r in range(data_length):
        for i in range(index.shape[1]):
            ref = index[0, i]
            target = index[1, i]
            length_2d_array[r, i, :] = data[r, ref:ref + 2] - data[r, target:target + 2]
    length_array = np.zeros((data_length, length_2d_array.shape[1]), dtype=np.float64)
    angle_array = np.zeros((data_length, length_2d_array.shape[1]), dtype=np.float64)
    for k in range(length_2d_array.shape[1]):
        for kk in range(data_length):
            length_array[kk, k] = np.linalg.norm(length_2d_array[kk, k, :])
            if kk < data_length - 1:
                try:
                    angle_array[kk, k] = np.rad2deg(
                        angle_between(length_2d_array[kk, k, :], length_2d_array[kk + 1, k, :]))
                except:
                    pass
    return length_array, angle_array


@jit(nopython=True)
def fast_smooth(data, n):
    data_boxcar_avg = np.zeros((data.shape[0], data.shape[1]))
    for body_part in range(data.shape[1]):
        data_boxcar_avg[:, body_part] = fast_running_mean(data[:, body_part], n)
    return data_boxcar_avg


@jit(nopython=True)
def fast_feature_extraction(data, framerate, index, smooth):
    window = int(np.round(0.05 / (1 / framerate)) * 2 - 1)
    features = List()
    for n in range(len(data)):
        displacement_raw = fast_displacment(data[n])
        length_raw, angle_raw = fast_length_angle(data[n], index)
        if smooth:
            displacement_run_mean = fast_smooth(displacement_raw, window)
            length_run_mean = fast_smooth(length_raw, window)
            angle_run_mean = fast_smooth(angle_raw, window)
            features.append(np.hstack((length_run_mean[1:, :], angle_run_mean[:-1, :], displacement_run_mean[:-1, :])))
        else:
            features.append(np.hstack((length_raw[:, :], angle_raw[:, :], displacement_raw[:, :])))
    return features


@jit(nopython=True)
def fast_feature_binning(features, framerate, index):
    binned_features_list = List()
    for n in range(len(features)):
        bin_width = int(framerate / 10)
        for s in range(bin_width):
            binned_features = np.zeros((int(features[n].shape[0] / bin_width), features[n].shape[1]), dtype=np.float64)
            for b in range(bin_width + s, features[n].shape[0], bin_width):
                binned_features[int(b / bin_width) - 1, 0:index.shape[1]] = np_mean(features[n][(b - bin_width):b,
                                                                                    0:index.shape[1]], 0)
                binned_features[int(b / bin_width) - 1, index.shape[1]:] = np.sum(features[n][(b - bin_width):b,
                                                                                  index.shape[1]:], axis=0)
            binned_features_list.append(binned_features)
    return binned_features_list


def bsoid_extract_numba(data, fps):
    smooth = False
    index = fast_nchoose2(int(data[0].shape[1] / 2), 2)
    features = fast_feature_extraction(data, fps, index * 2, smooth)
    binned_features = fast_feature_binning(features, fps, index * 2)
    return binned_features


def feature_extraction(train_datalist, num_train, framerate):
    f_integrated = []
    # for i in stqdm(range(num_train), desc="Extracting spatiotemporal features from pose"):
    for i in range(num_train):
        data_list = List()    
        data_list.append(train_datalist[i])
        binned_features = bsoid_extract_numba(data_list, framerate)
        f_integrated.append(binned_features[0])  # getting only the non-shifted
    features = np.vstack([f_integrated[m] for m in range(len(f_integrated))])
    scaler = StandardScaler()
    scaler.fit(features)
    scaled_features = scaler.transform(features)
    return features, scaled_features


def feature_extraction_with_extr_scaler(train_datalist, num_train, framerate, scaler):
    f_integrated = List()
    for i in range(num_train):
        binned_features = bsoid_extract_numba([train_datalist[i]], framerate)
        f_integrated.append(binned_features[0])  # getting only the non-shifted
    features = np.vstack([f_integrated[m] for m in range(len(f_integrated))])
    scaled_features = scaler.transform(features)
    return features, scaled_features


def unison_shuffled_copies(a, b, s):
    assert len(a) == len(b)
    np.random.seed(s)
    p = np.random.permutation(len(a))
    return a[p], b[p]
