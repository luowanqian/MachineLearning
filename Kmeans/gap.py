import numpy as np


def calculate_Wk(data, centroids, cluster):
    K = centroids.shape[0]
    wk = 0.0
    for k in range(K):
        data_in_cluster = data[cluster == k, :]
        center = centroids[k, :]
        num_points = data_in_cluster.shape[0]
        for i in range(num_points):
            wk = wk + np.linalg.norm(data_in_cluster[i, :]-center, ord=2) ** 2

    return wk


def bounding_box(data):
    dim = data.shape[1]
    boxes = []
    for i in range(dim):
        data_min = np.amin(data[:, i])
        data_max = np.amax(data[:, i])
        boxes.append((data_min, data_max))

    return boxes


def gap_statistic(data, max_K, B, cluster_algorithm):
    num_points, dim = data.shape
    K_range = np.arange(1, max_K, dtype=int)
    num_K = len(K_range)
    boxes = bounding_box(data)
    data_generate = np.zeros((num_points, dim))

    ''' 写法1
    log_Wks = np.zeros(num_K)
    gaps = np.zeros(num_K)
    sks = np.zeros(num_K)
    for ind_K, K in enumerate(K_range):
        cluster_centers, labels, _ = cluster_algorithm(data, K)
        log_Wks[ind_K] = np.log(calculate_Wk(data, cluster_centers, labels))

        # generate B reference data sets
        log_Wkbs = np.zeros(B)
        for b in range(B):
            for i in range(num_points):
                for j in range(dim):
                    data_generate[i][j] = \
                        np.random.uniform(boxes[j][0], boxes[j][1])
            cluster_centers, labels, _ = cluster_algorithm(data_generate, K)
            log_Wkbs[b] = \
                np.log(calculate_Wk(data_generate, cluster_centers, labels))
        gaps[ind_K] = np.mean(log_Wkbs) - log_Wks[ind_K]
        sks[ind_K] = np.std(log_Wkbs) * np.sqrt(1 + 1.0 / B)
    '''

    ''' 写法2
    '''
    log_Wks = np.zeros(num_K)
    for indK, K in enumerate(K_range):
        cluster_centers, labels, _ = cluster_algorithm(data, K)
        log_Wks[indK] = np.log(calculate_Wk(data, cluster_centers, labels))

    gaps = np.zeros(num_K)
    sks = np.zeros(num_K)
    log_Wkbs = np.zeros((B, num_K))

    # generate B reference data sets
    for b in range(B):
        for i in range(num_points):
            for j in range(dim):
                data_generate[i, j] = \
                    np.random.uniform(boxes[j][0], boxes[j][1])
        for indK, K in enumerate(K_range):
            cluster_centers, labels, _ = cluster_algorithm(data_generate, K)
            log_Wkbs[b, indK] = \
                np.log(calculate_Wk(data_generate, cluster_centers, labels))

    for k in range(num_K):
        gaps[k] = np.mean(log_Wkbs[:, k]) - log_Wks[k]
        sks[k] = np.std(log_Wkbs[:, k]) * np.sqrt(1 + 1.0 / B)

    return gaps, sks, log_Wks
