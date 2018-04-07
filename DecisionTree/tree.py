import numpy as np


def calculate_entropy(data, target):
    """
    Calculate the Shannon entropy of a dataset
    """
    num_entries = data.shape[0]
    label_counts = {}
    for i in range(num_entries):
        label = target[i]
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1

    entropy = 0.0
    for key in label_counts:
        prob = (1.0 * label_counts[key]) / num_entries
        entropy += -np.log2(prob) * prob

    return entropy


def split_dataset(data, target, axis, value):
    """
    Split data on a given feature
    """
    select_idx = data[:, axis] == value
    select_target = target[select_idx]
    select_data = data[select_idx]

    mask = np.ones(data.shape[1], dtype=bool)
    mask[axis] = False

    return select_data[:, mask], select_target


def choose_best_feature(data, target):
    """
    Choose the best feature to split on
    """
    num_data, num_features = data.shape
    if num_features == 1:
        return 0

    base_entropy = calculate_entropy(data, target)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_features):
        feature_vals = np.unique(data[:, i])
        new_entropy = 0.0
        for j in range(len(feature_vals)):
            subset_data, subset_target = \
                split_dataset(data, target, i, feature_vals[j])
            prob = float(len(subset_target)) / num_data
            new_entropy += \
                prob * calculate_entropy(subset_data, subset_target)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i

    return best_feature


def majority_class(target):
    class_list, counts = np.unique(target, return_counts=True)
    sort_idx = np.argsort(counts)

    return class_list[sort_idx[-1]]


def create_decision_tree(data, target, feature_labels,
                         feature_mappings, class_mappings):
    # stop when all classes are equal
    class_list = np.unique(target)
    if len(class_list) == 1:
        return class_mappings[class_list[0]]

    # when no more features, return majority
    if data.shape[0] == 0:
        return class_mappings[majority_class(target)]

    best_feature = choose_best_feature(data, target)
    best_feature_label = feature_labels[best_feature]
    del(feature_labels[best_feature])
    dt = {best_feature_label: {}}

    feature_vals = np.unique(data[:, best_feature])
    for i in range(len(feature_vals)):
        value = feature_vals[i]
        subset_feature_labels = feature_labels[:]
        subset_data, subset_target = \
            split_dataset(data, target, best_feature, value)
        dt[best_feature_label][feature_mappings[best_feature_label][value]] = \
            create_decision_tree(subset_data, subset_target,
                                 subset_feature_labels,
                                 feature_mappings, class_mappings)

    return dt
